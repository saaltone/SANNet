/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package demo;

import core.activation.ActivationFunction;
import core.layer.utils.AttentionLayerFactory;
import core.network.NeuralNetwork;
import core.network.NeuralNetworkConfiguration;
import core.network.NeuralNetworkException;
import core.layer.LayerType;
import core.optimization.OptimizationType;
import core.preprocess.ReadMIDI;
import utils.configurable.DynamicParamException;
import core.network.Persistence;
import utils.matrix.*;
import utils.sampling.BasicSampler;

import javax.sound.midi.Sequence;
import javax.sound.midi.Sequencer;
import java.util.*;

/**
 * Demo that synthesizes music by learning musical patterns from MIDI file.<br>
 * Uses recurrent neural network as basis to learn and synthesize music.<br>
 *
 */
public class Music {

    /**
     * Default constructor for music demo.
     *
     */
    public Music() {
    }

    /**
     * Main function that reads data, executes learning process and creates music based on learned patterns.
     *
     * @param args input arguments (not used).
     */
    public static void main(String[] args) {

        try {
            Music music = new Music();
            music.execute();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Function that learns musical patters from MIDI files and generates music based on learned model.
     *
     */
    private void execute() {
        try {

            int numberOfKeyInputs = 50;
            int numberOfVelocityInputs = 20;
            int numberOfTickInputs = 4;
            boolean encodeNoteOffs = false;
            long minTickDelta = 60;
            long maxTickDelta = 200;
            int maxEncodedTicks = 50;
            double tickScalingConstant = 0.65;
            int numberOfGeneratedSamples = 500;
            boolean useMultinomial = false;
            boolean prePlaySequence = true;
            boolean restoreNeuralNetwork = false;
            String path = "<PATH>/";
            ArrayList<String> fileNames = new ArrayList<>();
            fileNames.add(path + "Jesu-Joy-Of-Man-Desiring.mid");

            ReadMIDI readMIDI = new ReadMIDI();
            HashMap<Integer, HashMap<Integer, HashMap<Integer, Matrix>>> data = readMIDI.readFile(fileNames, numberOfKeyInputs, numberOfVelocityInputs, numberOfTickInputs, encodeNoteOffs, minTickDelta, maxTickDelta, maxEncodedTicks);
            ReadMIDI.Metadata metadata = readMIDI.getMetadata();

            if (prePlaySequence) {
                Sequence sequence = readMIDI.getSequenceAsMatrix(data.get(1).get(0), data.get(3).get(0), data.get(4).get(0), false, metadata, tickScalingConstant);
                readMIDI.play(sequence, 30, true);
            }

            String persistenceName = "<PATH>/MusicNN";

            NeuralNetwork neuralNetwork;
            if (!restoreNeuralNetwork) {
                int keyInputSize = data.get(0).get(0).get(0).getRows();
                int keyOutputSize = data.get(1).get(0).get(0).getRows();
                int velocityInputSize = data.get(2).get(0).get(0).getRows();
                int velocityOutputSize = data.get(3).get(0).get(0).getRows();
                int tickInputSize = data.get(4).get(0).get(0).getRows();
                int tickOutputSize = data.get(4).get(0).get(0).getRows();
                neuralNetwork = buildNeuralNetwork(numberOfKeyInputs, keyInputSize, keyOutputSize, numberOfVelocityInputs, velocityInputSize, velocityOutputSize, numberOfTickInputs, tickInputSize, tickOutputSize);
                neuralNetwork.setNeuralNetworkName("MIDI NN");
            }
            else {
                neuralNetwork = Persistence.restoreNeuralNetwork(persistenceName);
            }

            neuralNetwork.setAsClassification(true);

            Persistence persistence = new Persistence(true, 100, neuralNetwork, persistenceName, true);

            neuralNetwork.setPersistence(persistence);

            neuralNetwork.verboseTraining(10);

            neuralNetwork.setShowTrainingMetrics(true);

            neuralNetwork.start();

            neuralNetwork.print();
            neuralNetwork.printExpressions();
            neuralNetwork.printGradients();

            String params = "randomOrder = false, randomStart = false, stepSize = 1, shuffleSamples = false, sampleSize = 48, numberOfIterations = 100";
            HashMap<Integer, HashMap<Integer, Matrix>> trainingInputs = new HashMap<>();
            HashMap<Integer, HashMap<Integer, Matrix>> trainingOutputs = new HashMap<>();
            int trainInputPos = 0;
            int trainOutputPos = 0;
            for (int index = 0; index < numberOfKeyInputs + 1; index++) {
                if (index < numberOfKeyInputs) trainingInputs.put(trainInputPos++, data.get(0).get(index));
                else trainingOutputs.put(trainOutputPos++, data.get(1).get(index));
            }
            for (int index = 0; index < numberOfVelocityInputs + 1; index++) {
                if (index < numberOfVelocityInputs) trainingInputs.put(trainInputPos++, data.get(2).get(index));
                else trainingOutputs.put(trainOutputPos++, data.get(3).get(index));
            }
            for (int index = 0; index < numberOfTickInputs + 1; index++) {
                if (index < numberOfTickInputs) trainingInputs.put(trainInputPos++, data.get(4).get(index));
                else trainingOutputs.put(trainOutputPos++, data.get(4).get(index));
            }
            neuralNetwork.resetDependencies(false);
            neuralNetwork.setTrainingData(new BasicSampler(trainingInputs, trainingOutputs, params));

            int totalIterations = neuralNetwork.getTotalTrainingIterations();
            int fileVersion = 0;
            while (neuralNetwork.getTotalTrainingIterations() - totalIterations < 100000) {
                NeuralNetwork neuralNetworkForPrediction = neuralNetwork.copy();

                System.out.println("Training...");
                neuralNetwork.train(false, false);

                System.out.println("Predicting...");
                neuralNetworkForPrediction.start();

                HashMap<Integer, HashMap<Integer, Matrix>> result = new HashMap<>();
                result.put(0, new HashMap<>());
                result.put(1, new HashMap<>());
                result.put(2, new HashMap<>());

                TreeMap<Integer, Matrix> currentSample = new TreeMap<>();
                int predictInputPos = 0;
                for (int index = 0; index < numberOfKeyInputs; index++) currentSample.put(predictInputPos++, data.get(0).get(index).get(0));
                for (int index = 0; index < numberOfVelocityInputs; index++) currentSample.put(predictInputPos++, data.get(2).get(index).get(0));
                for (int index = 0; index < numberOfTickInputs; index++) currentSample.put(predictInputPos++, data.get(4).get(index).get(0));

                for (int sampleIndex = 0; sampleIndex < numberOfGeneratedSamples; sampleIndex++) {

                    TreeMap<Integer, Matrix> targetMatrices = predictNextSample(sampleIndex, neuralNetworkForPrediction, currentSample, result, useMultinomial);

                    int targetKey = targetMatrices.get(0).argmax()[0];
                    System.out.print("Key: " + targetKey + ", ");

                    int targetVelocity = targetMatrices.get(1).argmax()[0];
                    System.out.print("Velocity: " + targetVelocity + ", ");

                    int targetTick = targetMatrices.get(2).argmax()[0];
                    System.out.println("Tick: " + metadata.tickValueReverseMapping.get(targetTick));

                    getNextSample(currentSample, targetKey, targetVelocity, targetTick, numberOfKeyInputs, numberOfVelocityInputs, numberOfTickInputs, metadata);

                }
                neuralNetworkForPrediction.stop();

                System.out.println("Get MIDI sequence...");
                Sequence resultSequence = readMIDI.getSequence(result.get(0), result.get(1), result.get(2), metadata.resolution, false, tickScalingConstant);
                readMIDI.writeMIDI(resultSequence, path + "Result", ++fileVersion);
                System.out.println("Play MIDI...");
                Sequencer sequencer = readMIDI.play(resultSequence, 30, false);

                neuralNetwork.waitToComplete();

                System.out.println("Play MIDI complete...");
                readMIDI.stopPlaying(sequencer);
            }
            neuralNetwork.stop();
        }
        catch (Exception exception) {
            exception.printStackTrace();
            System.exit(-1);
        }
    }

    /**
     * Predicts next sample
     *
     * @param sampleIndex    sample index
     * @param neuralNetwork  neural network
     * @param currentSample  current sample
     * @param result         result
     * @param useMultinomial if true uses multinomial distribution in sampling.
     * @return next value
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    private TreeMap<Integer, Matrix> predictNextSample(int sampleIndex, NeuralNetwork neuralNetwork, TreeMap<Integer, Matrix> currentSample, HashMap<Integer, HashMap<Integer, Matrix>> result, boolean useMultinomial) throws NeuralNetworkException, MatrixException {
        TreeMap<Integer, Matrix> targetSample = neuralNetwork.predictMatrix(new TreeMap<>() {{ putAll(currentSample); }});
        for (Map.Entry<Integer, Matrix> entry : targetSample.entrySet()) {
            if (useMultinomial) targetSample.put(entry.getKey(), entry.getValue().getMultinomial(10));
            else targetSample.put(entry.getKey(), entry.getValue());
            result.get(entry.getKey()).put(sampleIndex,entry.getValue());
        }
        return targetSample;
    }

    /**
     * Returns next sample
     *
     * @param targetKey      target key
     * @param targetVelocity target velocity
     * @param targetTick     target tick
     * @param currentSample        current sample
     * @param metadata             metadata
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private void getNextSample(TreeMap<Integer, Matrix> currentSample, int targetKey, int targetVelocity, int targetTick, int numberOfKeyInputs, int numberOfVelocityInputs, int numberOfTickInputs, ReadMIDI.Metadata metadata) throws MatrixException {
        Matrix keyTargetMatrix = AbstractMatrix.encodeValueToBitColumnVector(targetKey, metadata.checkEncodeNoteOffs() ? 8 :7);
        Matrix velocityTargetMatrix = AbstractMatrix.encodeValueToBitColumnVector(targetVelocity, metadata.checkEncodeNoteOffs() ? 8 :7);
        Matrix tickTargetMatrix = DMatrix.getOneHotVector(metadata.numberOfEncodedTicks, targetTick);
        int offset = 0;
        for (int inputIndex = 0; inputIndex < numberOfKeyInputs; inputIndex++) {
            if (inputIndex < numberOfKeyInputs - 1) currentSample.put(offset + inputIndex, currentSample.get(offset + inputIndex + 1));
            else currentSample.put(offset + inputIndex, keyTargetMatrix);
        }
        offset += numberOfKeyInputs;
        for (int inputIndex = 0; inputIndex < numberOfVelocityInputs; inputIndex++) {
            if (inputIndex < numberOfVelocityInputs - 1) currentSample.put(offset + inputIndex, currentSample.get(offset + inputIndex + 1));
            else currentSample.put(offset + inputIndex, velocityTargetMatrix);
        }
        offset += numberOfKeyInputs + numberOfVelocityInputs;
        for (int inputIndex = 0; inputIndex < numberOfTickInputs; inputIndex++) {
            if (inputIndex < numberOfTickInputs - 1) currentSample.put(offset + inputIndex, currentSample.get(offset + inputIndex + 1));
            else currentSample.put(offset + inputIndex, tickTargetMatrix);
        }
    }

    /**
     * Builds decoder only transformer instance.
     *
     * @param numberOfKeyInputs      number of key inputs.
     * @param inputKeySize           input key size (digits as one hot encoded in sequence).
     * @param outputKeySize          output key size (digits as one hot encoded in sequence).
     * @param numberOfVelocityInputs number of velocity inputs.
     * @param inputVelocitySize      input velocity size (digits as one hot encoded in sequence).
     * @param outputVelocitySize     output velocity size (digits as one hot encoded in sequence).
     * @param numberOfTickInputs     number of tick inputs.
     * @param inputTickSize          input tick size (digits as one hot encoded in sequence).
     * @param outputTickSize         output tick size (digits as one hot encoded in sequence).
     * @return neural network instance.
     * @throws DynamicParamException  throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException        throws exception if custom function is attempted to be created with this constructor.
     */
    private static NeuralNetwork buildNeuralNetwork(int numberOfKeyInputs, int inputKeySize, int outputKeySize, int numberOfVelocityInputs, int inputVelocitySize, int outputVelocitySize, int numberOfTickInputs, int inputTickSize, int outputTickSize) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetworkConfiguration neuralNetworkConfiguration = new NeuralNetworkConfiguration();

        // Hyper parameters
        double tau = 0.2;
        double dropoutProbability = 0.1;
        boolean normalize = true;

        // Number of attention blocks for key, velocity and tick information.
        int keyAttentionBlocks = 4;
        int velocityAttentionBlocks = 2;
        int tickAttentionBlocks = 1;

        // Key neural network
        buildSingleNeuralNetwork(neuralNetworkConfiguration, numberOfKeyInputs, inputKeySize, outputKeySize, keyAttentionBlocks, tau, dropoutProbability, normalize);

        // Velocity neural network
        buildSingleNeuralNetwork(neuralNetworkConfiguration, numberOfVelocityInputs, inputVelocitySize, outputVelocitySize, velocityAttentionBlocks, tau, dropoutProbability, normalize);

        // Tick neural network
        buildSingleNeuralNetwork(neuralNetworkConfiguration, numberOfTickInputs, inputTickSize, outputTickSize, tickAttentionBlocks, tau, dropoutProbability, normalize);

        NeuralNetwork neuralNetwork = new NeuralNetwork(neuralNetworkConfiguration);

        neuralNetwork.setOptimizer(OptimizationType.RADAM);
        return neuralNetwork;
    }

    /**
     * Builds decoder transformer instance.
     *
     * @param numberOfInputs          number of inputs.
     * @param inputSize               input size (digits as one hot encoded in sequence).
     * @param outputSize              output size (digits as one hot encoded in sequence).
     * @param numberOfAttentionBlocks number of attention blocks.
     * @param tau                     temperature parameter for output softmax.
     * @param dropoutProbability      dropout probability.
     * @param normalize               if true includes normalization layers.
     * @throws DynamicParamException  throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException        throws exception if custom function is attempted to be created with this constructor.
     */
    private static void buildSingleNeuralNetwork(NeuralNetworkConfiguration neuralNetworkConfiguration, int numberOfInputs, int inputSize, int outputSize, int numberOfAttentionBlocks, double tau, double dropoutProbability, boolean normalize) throws DynamicParamException, NeuralNetworkException, MatrixException {
        // Transformer layers.
        int attentionLayer = AttentionLayerFactory.buildTransformer(neuralNetworkConfiguration, numberOfInputs, inputSize, 1, 1, numberOfAttentionBlocks, dropoutProbability, normalize, true);

        // Final feedforward layer.
        int hiddenLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX, "tau = " + tau), "width = " + outputSize);
        neuralNetworkConfiguration.connectLayers(attentionLayer, hiddenLayerIndex);

        // Output layer.
        int outputLayerIndex = neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
        neuralNetworkConfiguration.connectLayers(hiddenLayerIndex, outputLayerIndex);
    }

}
