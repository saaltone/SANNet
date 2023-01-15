/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package demo;

import core.activation.ActivationFunction;
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

            int numberOfInputs = 5;
            boolean encodeNoteOffs = false;
            long minTickDelta = 120;
            int maxEncodedTicks = 50;
            String path = "<PATH>/";
            ArrayList<String> fileNames = new ArrayList<>();
            fileNames.add(path + "Jesu-Joy-Of-Man-Desiring.mid");

            ReadMIDI readMIDI = new ReadMIDI();
            HashMap<Integer, HashMap<Integer, MMatrix>> data = readMIDI.readFile(fileNames, numberOfInputs, encodeNoteOffs, minTickDelta, maxEncodedTicks);
            ReadMIDI.Metadata metadata = readMIDI.getMetadata();

            Sequence sequence = readMIDI.getSequenceAsMMatrix(data.get((numberOfInputs + 1) - 1), data.get(2 * (numberOfInputs + 1) - 1), data.get(3 * (numberOfInputs + 1) - 1), false, metadata);
            readMIDI.play(sequence, 30, true);

            String persistenceName = "<PATH>/MusicNN";

            boolean restore = false;
            NeuralNetwork neuralNetwork;
            if (!restore) {
                int keyInputSize = data.get(0).get(0).get(0).getRows();
                int velocityInputSize = data.get((numberOfInputs + 1)).get(0).get(0).getRows();
                int tickInputSize = data.get(2 * (numberOfInputs + 1)).get(0).get(0).getRows();
                int keyOutputSize = data.get((numberOfInputs + 1) - 1).get(0).get(0).getRows();
                int velocityOutputSize = data.get(2 * (numberOfInputs + 1) - 1).get(0).get(0).getRows();
                int tickOutputSize = data.get(3 * (numberOfInputs + 1) - 1).get(0).get(0).getRows();
                neuralNetwork = buildNeuralNetwork(numberOfInputs, keyInputSize, numberOfInputs, velocityInputSize, numberOfInputs, tickInputSize, keyOutputSize, velocityOutputSize, tickOutputSize);
                neuralNetwork.setNeuralNetworkName("MIDI NN");
            }
            else {
                neuralNetwork = Persistence.restoreNeuralNetwork(persistenceName);
            }

            neuralNetwork.setAsClassification();

            Persistence persistence = new Persistence(true, 100, neuralNetwork, persistenceName, true);

            neuralNetwork.setPersistence(persistence);

            neuralNetwork.verboseTraining(10);

            neuralNetwork.start();

            neuralNetwork.print();
            neuralNetwork.printExpressions();
            neuralNetwork.printGradients();

            String params = "randomOrder = false, randomStart = false, stepSize = 1, shuffleSamples = false, sampleSize = 32, numberOfIterations = 100";
            HashMap<Integer, HashMap<Integer, MMatrix>> trainingInputs = new HashMap<>();
            HashMap<Integer, HashMap<Integer, MMatrix>> trainingOutputs = new HashMap<>();
            int trainInputPos = 0;
            int trainOutputPos = 0;
            for (int index = 0; index < 3; index++) {
                for (int index1 = 0; index1 < numberOfInputs + 1; index1++) {
                    if (index1 < numberOfInputs) trainingInputs.put(trainInputPos++, data.get(index * (numberOfInputs + 1) + index1));
                    else trainingOutputs.put(trainOutputPos++, data.get(index * (numberOfInputs + 1) + index1));
                }
            }
            neuralNetwork.resetDependencies(false);
            neuralNetwork.setTrainingData(new BasicSampler(trainingInputs, trainingOutputs, params));


            int totalIterations = neuralNetwork.getTotalIterations();
            int fileVersion = 0;
            while (neuralNetwork.getTotalIterations() - totalIterations < 100000) {
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
                for (int index = 0; index < 3; index++) {
                    for (int index1 = 0; index1 < numberOfInputs + 1; index1++) {
                        if (index1 < numberOfInputs) currentSample.put(predictInputPos++, data.get(index * (numberOfInputs + 1) + index1).get(0).get(0));
                    }
                }

                for (int sampleIndex = 0; sampleIndex < 300; sampleIndex++) {

                    TreeMap<Integer, Matrix> targetMatrices = predictNextSample(sampleIndex, neuralNetworkForPrediction, currentSample, result);
                    int targetKey = targetMatrices.get(0).argmax()[0];
                    System.out.print("Key: " + metadata.decodeItem(targetKey, metadata.minKeyValue) + ", ");
                    int targetVelocity = targetMatrices.get(1).argmax()[0];
                    System.out.print("Velocity: " + metadata.decodeItem(targetVelocity, metadata.minVelocityValue) + ", ");
                    int targetTick = targetMatrices.get(2).argmax()[0];
                    System.out.println("Tick: " + metadata.tickValueReverseMapping.get(targetTick));

                    Matrix keyTargetMatrix = new DMatrix(metadata.getKeyOutputSize(), 1);
                    keyTargetMatrix.setValue(targetKey, 0, 1);
                    Matrix velocityTargetMatrix = new DMatrix(metadata.getVelocityOutputSize(), 1);
                    velocityTargetMatrix.setValue(targetVelocity, 0, 1);
                    Matrix tickTargetMatrix = new DMatrix(metadata.numberOfEncodedTicks, 1);
                    tickTargetMatrix.setValue(targetTick, 0, 1);

                    getNextSample(currentSample, keyTargetMatrix, velocityTargetMatrix, tickTargetMatrix, numberOfInputs);

                }
                neuralNetworkForPrediction.stop();

                System.out.println("Get MIDI sequence...");
                Sequence resultSequence = readMIDI.getSequence(result.get(0), result.get(1), result.get(2), metadata.resolution, false);
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
     * @param sampleIndex sample index
     * @param neuralNetwork neural network
     * @param currentSample current sample
     * @param result result
     * @return next value
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    private TreeMap<Integer, Matrix> predictNextSample(int sampleIndex, NeuralNetwork neuralNetwork, TreeMap<Integer, Matrix> currentSample, HashMap<Integer, HashMap<Integer, Matrix>> result) throws MatrixException, NeuralNetworkException {
        TreeMap<Integer, Matrix> targetSample = neuralNetwork.predictMatrix(new TreeMap<>() {{ putAll(currentSample); }});
        for (Map.Entry<Integer, Matrix> entry : targetSample.entrySet()) result.get(entry.getKey()).put(sampleIndex,entry.getValue());
        return targetSample;
    }

    /**
     * Returns next sample
     *
     * @param currentSample current sample
     * @param keyTargetMatrix key target matrix
     * @param velocityTargetMatrix velocity target matrix
     * @param tickTargetMatrix tick target matrix
     */
    private void getNextSample(TreeMap<Integer, Matrix> currentSample, Matrix keyTargetMatrix, Matrix velocityTargetMatrix, Matrix tickTargetMatrix, int numberOfInputs) {
        for (int index = 0; index < 3; index++) {
            int offset = index * numberOfInputs;
            for (int index1 = 0; index1 < numberOfInputs; index1++) {
                if (index1 < numberOfInputs - 1) currentSample.put(offset + index1, currentSample.get(offset + index1 + 1));
                else currentSample.put(offset + index1, index == 0 ? keyTargetMatrix : index == 1 ? velocityTargetMatrix : tickTargetMatrix);
            }
        }
    }


    /**
     * Builds recurrent neural network (GRU) instance.
     *
     * @param inputKeySize input key size (digits as one hot encoded in sequence).
     * @param inputVelocitySize input velocity size (digits as one hot encoded in sequence).
     * @param inputTickSize input tick size (digits as one hot encoded in sequence).
     * @param outputKeySize output key size (digits as one hot encoded in sequence).
     * @param outputVelocitySize output velocity size (digits as one hot encoded in sequence).
     * @param outputTickSize output tick size (digits as one hot encoded in sequence).
     * @return neural network instance.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static NeuralNetwork buildNeuralNetwork(int numberOfKeyInputs, int inputKeySize, int numberOfVelocityInputs, int inputVelocitySize, int numberOfTickInputs, int inputTickSize, int outputKeySize, int outputVelocitySize, int outputTickSize) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetworkConfiguration neuralNetworkConfiguration = new NeuralNetworkConfiguration();

        // Encoder and attention layers for processing key value information.
        int keyAttentionLayerIndex = buildAttentionModule(neuralNetworkConfiguration, inputKeySize, inputKeySize, numberOfKeyInputs, 1);
        // Encoder and attention layers for processing velocity value information.
        int velocityAttentionLayerIndex = buildAttentionModule(neuralNetworkConfiguration, inputVelocitySize, inputVelocitySize, numberOfVelocityInputs, 1);
        // Encoder and attention layers for processing tick value information.
        int tickAttentionLayerIndex = buildAttentionModule(neuralNetworkConfiguration, inputTickSize, inputTickSize, numberOfTickInputs, 1);

        // Decoder modules for key, velocity and tick information
//        int keyDecoderIndex = buildDecoderModule(neuralNetworkConfiguration, keyAttentionLayerIndex, outputKeySize, true);
//        int velocityDecoderIndex = buildDecoderModule(neuralNetworkConfiguration, velocityAttentionLayerIndex, outputVelocitySize, true);
//        int tickDecoderIndex = buildDecoderModule(neuralNetworkConfiguration, tickAttentionLayerIndex, outputTickSize, true);

        // Final feedforward layers for key, velocity and tick information.
        int keyHiddenLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX), "width = " + outputKeySize);
        int velocityHiddenLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX), "width = " + outputVelocitySize);
        int tickHiddenLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX), "width = " + outputTickSize);
        neuralNetworkConfiguration.connectLayers(keyAttentionLayerIndex, keyHiddenLayerIndex);
        neuralNetworkConfiguration.connectLayers(velocityAttentionLayerIndex, velocityHiddenLayerIndex);
        neuralNetworkConfiguration.connectLayers(tickAttentionLayerIndex, tickHiddenLayerIndex);

        // Output layers for key, velocity and tick information.
        int keyOutputLayerIndex = neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
        int velocityOutputLayerIndex = neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
        int tickOutputLayerIndex = neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
        neuralNetworkConfiguration.connectLayers(keyHiddenLayerIndex, keyOutputLayerIndex);
        neuralNetworkConfiguration.connectLayers(velocityHiddenLayerIndex, velocityOutputLayerIndex);
        neuralNetworkConfiguration.connectLayers(tickHiddenLayerIndex, tickOutputLayerIndex);

        NeuralNetwork neuralNetwork = new NeuralNetwork(neuralNetworkConfiguration);

        neuralNetwork.setOptimizer(OptimizationType.RADAM);
        return neuralNetwork;
    }

    /**
     * Builds bi-directional RNN module.
     *
     * @param neuralNetworkConfiguration neural network configuration.
     * @param inputWidth input width
     * @param hiddenWidth hidden width
     * @param numberOfUnits number of units
     * @return index of module output layer.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static int buildInputEncoderModule(NeuralNetworkConfiguration neuralNetworkConfiguration, int inputWidth, int hiddenWidth, int numberOfUnits, boolean biDirectional) throws NeuralNetworkException, DynamicParamException, MatrixException {
        int inputLayerIndex = neuralNetworkConfiguration.addInputLayer("width = " + inputWidth);
        if (biDirectional) {
            int[] hiddenLayerIndices = new int[2 * numberOfUnits];
            for (int index = 0; index < numberOfUnits; index++) {
                hiddenLayerIndices[2 * index] = neuralNetworkConfiguration.addHiddenLayer(LayerType.MINGRU, "width = " + hiddenWidth);
                hiddenLayerIndices[2 * index + 1] = neuralNetworkConfiguration.addHiddenLayer(LayerType.MINGRU, "width = " + hiddenWidth + ", reversedInput = true");
                neuralNetworkConfiguration.connectLayers(inputLayerIndex, hiddenLayerIndices[2 * index]);
                neuralNetworkConfiguration.connectLayers(inputLayerIndex, hiddenLayerIndices[2 * index + 1]);
            }
            int joinLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.JOIN);
            for (int layerIndex : hiddenLayerIndices) {
                neuralNetworkConfiguration.connectLayers(layerIndex, joinLayerIndex);
            }
            return joinLayerIndex;
        }
        else {
            int[] hiddenLayerIndices = new int[numberOfUnits];
            for (int index = 0; index < numberOfUnits; index++) {
                hiddenLayerIndices[index] = neuralNetworkConfiguration.addHiddenLayer(LayerType.MINGRU, "width = " + hiddenWidth);
                neuralNetworkConfiguration.connectLayers(inputLayerIndex, hiddenLayerIndices[index]);
            }
            if (hiddenLayerIndices.length > 1) {
                int joinLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.JOIN);
                for (int layerIndex : hiddenLayerIndices) {
                    neuralNetworkConfiguration.connectLayers(layerIndex, joinLayerIndex);
                }
                return joinLayerIndex;
            }
            else return hiddenLayerIndices[0];
        }
    }

    /**
     * Builds attention module
     *
     * @param neuralNetworkConfiguration neural network configuration
     * @param inputSize input size
     * @param hiddenSize hidden size
     * @param numberOfInputs number of inputs
     * @param numberOfUnits number of units
     * @return attention layer index
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static int buildAttentionModule(NeuralNetworkConfiguration neuralNetworkConfiguration, int inputSize, int hiddenSize, int numberOfInputs, int numberOfUnits) throws MatrixException, NeuralNetworkException, DynamicParamException {
        // Encoder layers for input information.
        int[] encoderIndices = new int[numberOfInputs];
        for (int inputIndex = 0; inputIndex < numberOfInputs; inputIndex++) {
            encoderIndices[inputIndex] = buildInputEncoderModule(neuralNetworkConfiguration, inputSize, hiddenSize, numberOfUnits, true);
        }

        // Attention layer for input information.
        int combinedIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.ATTENTION);
        for (int inputIndex = 0; inputIndex < numberOfInputs; inputIndex++) {
            neuralNetworkConfiguration.connectLayers(encoderIndices[inputIndex], combinedIndex);
        }
//        return combinedIndex;

        // Normalization layer for input information.
/*        int normLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.LAYER_NORMALIZATION);
        neuralNetworkConfiguration.connectLayers(combinedIndex, normLayerIndex);
        return normLayerIndex; */

        // Regularization layer for input information.
        int regularizationLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.DROPOUT, "probability = 0.1");
        neuralNetworkConfiguration.connectLayers(combinedIndex, regularizationLayerIndex);
        return regularizationLayerIndex;
    }

    /**
     * Builds input module.
     *
     * @param neuralNetworkConfiguration neural network configuration.
     * @param previousLayerIndex previous layer index
     * @param hiddenWidth hidden width
     * @return index of module output layer.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static int buildDecoderModule(NeuralNetworkConfiguration neuralNetworkConfiguration, int previousLayerIndex, int hiddenWidth, boolean asBidirectional) throws NeuralNetworkException, DynamicParamException, MatrixException {
        if (asBidirectional) {
            int hiddenLayerIndex1 = neuralNetworkConfiguration.addHiddenLayer(LayerType.MINGRU, "width = " + hiddenWidth);
            int hiddenLayerIndex2 = neuralNetworkConfiguration.addHiddenLayer(LayerType.MINGRU, "width = " + hiddenWidth + ", reversedInput = true");
            int hiddenLayerIndex3 = neuralNetworkConfiguration.addHiddenLayer(LayerType.JOIN, "width = " + hiddenWidth);
            int hiddenLayerIndex4 = neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.SOFTMAX));
            neuralNetworkConfiguration.connectLayers(previousLayerIndex, hiddenLayerIndex1);
            neuralNetworkConfiguration.connectLayers(previousLayerIndex, hiddenLayerIndex2);
            neuralNetworkConfiguration.connectLayers(hiddenLayerIndex1, hiddenLayerIndex3);
            neuralNetworkConfiguration.connectLayers(hiddenLayerIndex2, hiddenLayerIndex3);
            neuralNetworkConfiguration.connectLayers(hiddenLayerIndex3, hiddenLayerIndex4);
            return hiddenLayerIndex4;
        }
        else {
            int hiddenLayerIndex1 = neuralNetworkConfiguration.addHiddenLayer(LayerType.GRU, "width = " + hiddenWidth);
            int hiddenLayerIndex2 = neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.SOFTMAX));
            neuralNetworkConfiguration.connectLayers(previousLayerIndex, hiddenLayerIndex1);
            neuralNetworkConfiguration.connectLayers(hiddenLayerIndex1, hiddenLayerIndex2);
            return hiddenLayerIndex2;
        }
    }

}
