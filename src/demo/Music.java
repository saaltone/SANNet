/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package demo;

import core.network.NeuralNetwork;
import core.network.NeuralNetworkConfiguration;
import core.network.NeuralNetworkException;
import core.activation.ActivationFunction;
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

            int numOfInputs = 5;
            boolean encodeNoteOffs = false;
            long minTickDelta = 80;
            int maxEncodedTicks = 50;
            String path = "<PATH>/";
            ArrayList<String> fileNames = new ArrayList<>();
            fileNames.add(path + "Jesu-Joy-Of-Man-Desiring.mid");

            ReadMIDI readMIDI = new ReadMIDI();
            HashMap<Integer, HashMap<Integer, MMatrix>> data = readMIDI.readFile(fileNames, numOfInputs, encodeNoteOffs, minTickDelta, maxEncodedTicks);
            ReadMIDI.Metadata metadata = readMIDI.getMetadata();

            Sequence sequence = readMIDI.getSequenceAsMMatrix(data.get(1), data.get(3), data.get(5), false, metadata);
            readMIDI.play(sequence, 30, true);

            String persistenceName = "<PATH>/MusicNN";

            boolean restore = false;
            NeuralNetwork neuralNetwork;
            if (!restore) {
                neuralNetwork = buildNeuralNetwork(data.get(0).get(0).get(0).getRows(), data.get(2).get(0).get(0).getRows(), data.get(4).get(0).get(0).getRows(), data.get(1).get(0).get(0).getRows(), data.get(3).get(0).get(0).getRows(), data.get(5).get(0).get(0).getRows(), data.get(0).get(0).get(0).getRows() * 4);
                neuralNetwork.setNeuralNetworkName("MIDI NN");
            }
            else {
                neuralNetwork = Persistence.restoreNeuralNetwork(persistenceName);
            }

            neuralNetwork.setAsClassification();

            Persistence persistenceCombined = new Persistence(true, 100, neuralNetwork, persistenceName, true);

            neuralNetwork.setPersistence(persistenceCombined);

            neuralNetwork.verboseTraining(10);

            neuralNetwork.start();

            neuralNetwork.print();
            neuralNetwork.printExpressions();
            neuralNetwork.printGradients();

            String params = "randomOrder = false, randomStart = false, stepSize = 1, shuffleSamples = false, sampleSize = 48, numberOfIterations = 100";
            neuralNetwork.setTrainingData(new BasicSampler(new HashMap<>() {{ put(0, data.get(0)); put(1, data.get(2)); put(2, data.get(4)); }}, new HashMap<>() {{ put(0, data.get(1)); put(1, data.get(3)); put(2, data.get(5)); }}, params));

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
                currentSample.put(0, data.get(0).get(0).get(0));
                currentSample.put(1, data.get(2).get(0).get(0));
                currentSample.put(2, data.get(4).get(0).get(0));

                for (int sampleIndex = 0; sampleIndex < 300; sampleIndex++) {

                    TreeMap<Integer, Matrix> targetMatrices = predictNextSample(sampleIndex, neuralNetworkForPrediction, currentSample, result);
                    int targetKey = targetMatrices.get(0).argmax()[0];
                    System.out.print("Key: " + metadata.decodeItem(targetKey, metadata.minKeyValue) + ", ");
                    int targetVelocity = targetMatrices.get(1).argmax()[0];
                    System.out.print("Velocity: " + metadata.decodeItem(targetVelocity, metadata.minVelocityValue) + ", ");
                    int targetTick = targetMatrices.get(2).argmax()[0];
                    System.out.println("Tick: " + metadata.tickValueReverseMapping.get(targetTick));

                    currentSample.put(0, getNextSample(currentSample.get(0), ComputableMatrix.encodeToBitColumnVector(targetKey, ComputableMatrix.numberOfBits(metadata.getKeyOutputSize()))));
                    currentSample.put(1, getNextSample(currentSample.get(1), ComputableMatrix.encodeToBitColumnVector(targetVelocity, ComputableMatrix.numberOfBits(metadata.getVelocityOutputSize()))));
                    currentSample.put(2, getNextSample(currentSample.get(2), ComputableMatrix.encodeToBitColumnVector(targetTick, ComputableMatrix.numberOfBits(metadata.numberOfEncodedTicks))));

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
     * @param targetMatrix target matrix
     * @return next sample
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix getNextSample(Matrix currentSample, Matrix targetMatrix) throws MatrixException {
        ArrayList<Matrix> currentSamples = currentSample.getSubMatrices();
        int maxSampleIndex = currentSamples.size() - 1;
        for (int sampleIndex = 0; sampleIndex < maxSampleIndex; sampleIndex++) currentSamples.set(sampleIndex, currentSamples.get(sampleIndex + 1));
        currentSamples.set(maxSampleIndex, targetMatrix);
        return new JMatrix(currentSamples, true);
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
     * @param hiddenSize hidden layer size of neural network.
     * @return neural network instance.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static NeuralNetwork buildNeuralNetwork(int inputKeySize, int inputVelocitySize, int inputTickSize, int outputKeySize, int outputVelocitySize, int outputTickSize, int hiddenSize) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();

        NeuralNetworkConfiguration neuralNetworkConfiguration = new NeuralNetworkConfiguration();

        // Layers for processing key value information.
        int hiddenKeySize = 2 * inputKeySize;
        neuralNetworkConfiguration.addInputLayer("width = " + inputKeySize);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.GRU, "width = " + hiddenKeySize);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.GRU, "width = " + hiddenKeySize);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.GRU, "width = " + hiddenKeySize + ", reversedInput = true");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.GRU, "width = " + hiddenKeySize + ", reversedInput = true");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.CONNECTOR, "joinInputs = true");
        neuralNetworkConfiguration.connectLayers(0, 1);
        neuralNetworkConfiguration.connectLayers(0, 3);
        neuralNetworkConfiguration.connectLayers(1, 2);
        neuralNetworkConfiguration.connectLayers(3, 4);
        neuralNetworkConfiguration.connectLayers(2, 5);
        neuralNetworkConfiguration.connectLayers(4, 5);
        neuralNetworkConfiguration.connectLayers(0, 5);

        // Layers for processing velocity value information.
        int hiddenVelocitySize = 2 * inputVelocitySize;
        neuralNetworkConfiguration.addInputLayer("width = " + inputVelocitySize);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.GRU, "width = " + hiddenVelocitySize);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.GRU, "width = " + hiddenVelocitySize + ", reversedInput = true");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.CONNECTOR, "joinInputs = true");
        neuralNetworkConfiguration.connectLayers(6, 7);
        neuralNetworkConfiguration.connectLayers(6, 8);
        neuralNetworkConfiguration.connectLayers(7, 9);
        neuralNetworkConfiguration.connectLayers(8, 9);
        neuralNetworkConfiguration.connectLayers(6, 9);

        // Layers for processing tick value information.
        int hiddenTickSize = 2 * inputTickSize;
        neuralNetworkConfiguration.addInputLayer("width = " + inputTickSize);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.GRU, "width = " + hiddenTickSize);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.GRU, "width = " + hiddenTickSize + ", reversedInput = true");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.CONNECTOR, "joinInputs = true");
        neuralNetworkConfiguration.connectLayers(10, 11);
        neuralNetworkConfiguration.connectLayers(10, 12);
        neuralNetworkConfiguration.connectLayers(11, 13);
        neuralNetworkConfiguration.connectLayers(12, 13);
        neuralNetworkConfiguration.connectLayers(10, 13);

        // Layers connecting key, value and tick value information.
        int hiddenConnectSize = hiddenKeySize + hiddenVelocitySize + hiddenTickSize;
        neuralNetworkConfiguration.addHiddenLayer(LayerType.CONNECTOR, "width = " + hiddenConnectSize);
        neuralNetworkConfiguration.connectLayers(5, 14);
        neuralNetworkConfiguration.connectLayers(9, 14);
        neuralNetworkConfiguration.connectLayers(13, 14);

        neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX), "width = " + outputKeySize);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX), "width = " + outputVelocitySize);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX), "width = " + outputTickSize);
        neuralNetworkConfiguration.connectLayers(14, 15);
        neuralNetworkConfiguration.connectLayers(14, 16);
        neuralNetworkConfiguration.connectLayers(14, 17);

        neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
        neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
        neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
        neuralNetworkConfiguration.connectLayers(15, 18);
        neuralNetworkConfiguration.connectLayers(16, 19);
        neuralNetworkConfiguration.connectLayers(17, 20);

        neuralNetwork.build(neuralNetworkConfiguration);
        neuralNetwork.setOptimizer(OptimizationType.RADAM);
        return neuralNetwork;
    }

}
