/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package demo;

import core.network.NeuralNetwork;
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
            boolean excludeZeroValuedEntries = true;
            String path = "<PATH>/";
            ArrayList<String> fileNames = new ArrayList<>();
            fileNames.add(path + "howdsolo.mid");

            ReadMIDI readMIDI = new ReadMIDI();
            HashMap<Integer, HashMap<Integer, MMatrix>> data = readMIDI.readFile(fileNames, numOfInputs, encodeNoteOffs, excludeZeroValuedEntries, 8);
            ReadMIDI.Metadata metadata = readMIDI.getMetadata();

            Sequence sequence = readMIDI.getSequence(data.get(1), data.get(3), data.get(5), metadata.resolution, false, encodeNoteOffs, metadata);
            readMIDI.play(sequence, 10, true);

            String persistenceNameKey = path + "MusicNNKey";
            String persistenceNameVelocity = path + "MusicNNVelocity";
            String persistenceNameTick = path + "MusicNNTick";

            boolean restore = false;
            NeuralNetwork neuralNetworkKey;
            NeuralNetwork neuralNetworkVelocity;
            NeuralNetwork neuralNetworkTick;
            if (!restore) {
                neuralNetworkKey = buildNeuralNetwork(data.get(0).get(0).get(0).getRows(), data.get(1).get(0).get(0).getRows(), data.get(0).get(0).get(0).getRows() * 5, 0);
                neuralNetworkKey.setNeuralNetworkName("MIDI_key NN");
                neuralNetworkVelocity = buildNeuralNetwork(data.get(2).get(0).get(0).getRows(), data.get(3).get(0).get(0).getRows(), data.get(2).get(0).get(0).getRows() * 4, 1);
                neuralNetworkVelocity.setNeuralNetworkName("MIDI_velocity NN");
                neuralNetworkTick = buildNeuralNetwork(data.get(4).get(0).get(0).getRows(), data.get(5).get(0).get(0).getRows(), data.get(4).get(0).get(0).getRows() * 3, 2);
                neuralNetworkTick.setNeuralNetworkName("MIDI_tick NN");
            }
            else {
                neuralNetworkKey = Persistence.restoreNeuralNetwork(persistenceNameKey);
                neuralNetworkVelocity = Persistence.restoreNeuralNetwork(persistenceNameVelocity);
                neuralNetworkTick = Persistence.restoreNeuralNetwork(persistenceNameTick);
            }

            neuralNetworkKey.setAsClassification();
            neuralNetworkVelocity.setAsClassification();
            neuralNetworkTick.setAsClassification();

            Persistence persistenceKey = new Persistence(true, 100, neuralNetworkKey, persistenceNameKey, true);
            Persistence persistenceVelocity = new Persistence(true, 100, neuralNetworkVelocity, persistenceNameVelocity, true);
            Persistence persistenceTick = new Persistence(true, 100, neuralNetworkTick, persistenceNameTick, true);

            neuralNetworkKey.setPersistence(persistenceKey);
            neuralNetworkVelocity.setPersistence(persistenceVelocity);
            neuralNetworkTick.setPersistence(persistenceTick);

            neuralNetworkKey.verboseTraining(10);
            neuralNetworkVelocity.verboseTraining(10);
            neuralNetworkTick.verboseTraining(10);

            neuralNetworkKey.start();
            neuralNetworkVelocity.start();
            neuralNetworkTick.start();

            neuralNetworkKey.print();
            neuralNetworkVelocity.print();
            neuralNetworkTick.print();

            String keyParams = "randomOrder = false, randomStart = false, stepSize = 1, shuffleSamples = false, sampleSize = 48, numberOfIterations = 100";
            neuralNetworkKey.setTrainingData(new BasicSampler(data.get(0), data.get(1),keyParams));
            String velocityParams = "randomOrder = false, randomStart = false, stepSize = 1, shuffleSamples = false, sampleSize = 48, numberOfIterations = 100";
            neuralNetworkVelocity.setTrainingData(new BasicSampler(data.get(2), data.get(3),velocityParams));
            String tickParams = "randomOrder = false, randomStart = false, stepSize = 1, shuffleSamples = false, sampleSize = 48, numberOfIterations = 100";
            neuralNetworkTick.setTrainingData(new BasicSampler(data.get(4), data.get(5),tickParams));

            int totalIterations = neuralNetworkKey.getTotalIterations();
            int fileVersion = 0;
            while (neuralNetworkKey.getTotalIterations() - totalIterations < 100000) {
                NeuralNetwork neuralNetworkKeyForPrediction = neuralNetworkKey.copy();
                NeuralNetwork neuralNetworkVelocityForPrediction = neuralNetworkVelocity.copy();
                NeuralNetwork neuralNetworkTickForPrediction = neuralNetworkTick.copy();

                System.out.println("Training...");
                neuralNetworkKey.train(false, false);
                neuralNetworkVelocity.train(false, false);
                neuralNetworkTick.train(false, false);

                System.out.println("Predicting...");
                neuralNetworkKeyForPrediction.start();
                neuralNetworkVelocityForPrediction.start();
                neuralNetworkTickForPrediction.start();

                HashMap<Integer, MMatrix> resultKey = new HashMap<>();
                HashMap<Integer, MMatrix> resultVelocity = new HashMap<>();
                HashMap<Integer, MMatrix> resultTick = new HashMap<>();

                Matrix currentSampleKey = data.get(0).get(0).get(0);
                Matrix currentSampleVelocity = data.get(2).get(0).get(0);
                Matrix currentSampleTick = data.get(4).get(0).get(0);

                for (int sampleIndex = 0; sampleIndex < 300; sampleIndex++) {
                    Matrix targetKeyMatrix = predictNextSample(sampleIndex, neuralNetworkKeyForPrediction, currentSampleKey, resultKey);
                    int targetKey = targetKeyMatrix.argmax()[0];
                    System.out.print("Key: " + (metadata.minKeyValue + targetKey) + ", ");

                    Matrix targetVelocityMatrix = predictNextSample(sampleIndex, neuralNetworkVelocityForPrediction, currentSampleVelocity, resultVelocity);
                    int targetVelocity = targetVelocityMatrix.argmax()[0];
                    System.out.print("Velocity: " + (metadata.minVelocityValue + targetVelocity) + ", ");

                    Matrix targetTickMatrix = predictNextSample(sampleIndex, neuralNetworkTickForPrediction, currentSampleTick, resultTick);
                    int targetTick = targetTickMatrix.argmax()[0];
                    System.out.println("Tick: " + metadata.tickValueReverseMapping.get(targetTick));

                    currentSampleKey = getNextSample(currentSampleKey, ComputableMatrix.encodeToBitColumnVector(targetKey, metadata.keyBitVectorSize));
                    currentSampleVelocity = getNextSample(currentSampleVelocity, ComputableMatrix.encodeToBitColumnVector(targetVelocity, metadata.velocityBitVectorSize));
                    currentSampleTick = getNextSample(currentSampleTick, targetTickMatrix);

                }
                neuralNetworkKeyForPrediction.stop();
                neuralNetworkVelocityForPrediction.stop();
                neuralNetworkTickForPrediction.stop();

                System.out.println("Get MIDI sequence...");
                Sequence resultSequence = readMIDI.getSequence(resultKey, resultVelocity, resultTick, metadata.resolution, false, encodeNoteOffs, metadata);
                readMIDI.writeMIDI(resultSequence, path + "Result", ++fileVersion);
                System.out.println("Play MIDI...");
                Sequencer sequencer = readMIDI.play(resultSequence, 30, false);

                neuralNetworkKey.waitToComplete();
                neuralNetworkVelocity.waitToComplete();
                neuralNetworkTick.waitToComplete();

                System.out.println("Play MIDI complete...");
                readMIDI.stopPlaying(sequencer);
            }
            neuralNetworkKey.stop();
            neuralNetworkVelocity.stop();
            neuralNetworkTick.stop();
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
    private Matrix predictNextSample(int sampleIndex, NeuralNetwork neuralNetwork, Matrix currentSample, HashMap<Integer, MMatrix> result) throws MatrixException, NeuralNetworkException {
        Matrix targetSample = neuralNetwork.predict(currentSample);
        result.put(sampleIndex, new MMatrix(targetSample));
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
     * @param inputSize input size of neural network (digits as one hot encoded in sequence).
     * @param outputSize output size of neural network (digits as one hot encoded in sequence).
     * @param hiddenSize hidden layer size of neural network.
     * @param neuralNetworkType neural network type (key, velocity or tick neural network).
     * @return neural network instance.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize, int hiddenSize, int neuralNetworkType) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = " + inputSize);
        neuralNetwork.addHiddenLayer(LayerType.BILSTM, "width = " + hiddenSize);
        switch (neuralNetworkType) {
            case 0 -> {
                neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX), "width = " + outputSize);
                neuralNetwork.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
            }
            case 1 -> {
                neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX), "width = " + outputSize);
                neuralNetwork.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
            }
            case 2 -> {
                neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX), "width = " + outputSize);
                neuralNetwork.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
            }
            default -> {
                System.out.println("Unknown loss type.");
                System.exit(-1);
            }
        }
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.RADAM);
        return neuralNetwork;
    }

}
