/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package demo;

import core.NeuralNetwork;
import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.metrics.MetricsType;
import core.optimization.OptimizationType;
import core.preprocess.ReadMIDI;
import utils.DynamicParamException;
import utils.Persistence;
import utils.matrix.*;
import utils.sampling.BasicSampler;

import javax.sound.midi.Sequence;
import javax.sound.midi.Sequencer;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;

/**
 * Demo that synthesizes music by learning musical patterns from MIDI file.<br>
 * Uses recurrent neural network as basis to learn and synthesize music.
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
            NeuralNetwork neuralNetworkKey;
            NeuralNetwork neuralNetworkVelocity;
            NeuralNetwork neuralNetworkTick;

            String path = "<PATH>/";
            HashSet<String> fileNames = new HashSet<>();
            fileNames.add(path + "canon4.mid");
            HashMap<Integer, LinkedHashMap<Integer, MMatrix>> data = ReadMIDI.readFile(fileNames);
            long scalingFactor = ReadMIDI.scaleTickData(data.get(4), data.get(5), 250);
            float divisionType = 0;
            int resolution = 0;
            for (String fileNme : fileNames) {
                divisionType += ReadMIDI.getDivisionType(fileNme);
                resolution += ReadMIDI.getResolution(fileNme);
            }
            divisionType /= (double)fileNames.size();
            resolution /= (double)fileNames.size();

            Sequence sequence = ReadMIDI.getSequence(data.get(0), data.get(2), data.get(4), divisionType, resolution, scalingFactor);
            ReadMIDI.play(sequence, 10, true);

            String persistenceNameKey = "<PATH>/MusicNNKey";
            String persistenceNameVelocity = "<PATH>/MusicNNVelocity";
            String persistenceNameTick = "<PATH>/MusicNNTick";

            boolean restore = false;
            if (!restore) {
                neuralNetworkKey = buildNeuralNetwork(data.get(0).get(0).get(0).getRows(), data.get(1).get(0).get(0).getRows(), data.get(0).get(0).get(0).getRows(), 0);
                neuralNetworkKey.setNeuralNetworkName("MIDI_key NN");
                neuralNetworkVelocity = buildNeuralNetwork(data.get(2).get(0).get(0).getRows(), data.get(3).get(0).get(0).getRows(), data.get(2).get(0).get(0).getRows(), 1);
                neuralNetworkVelocity.setNeuralNetworkName("MIDI_velocity NN");
                neuralNetworkTick = buildNeuralNetwork(data.get(4).get(0).get(0).getRows(), data.get(5).get(0).get(0).getRows(), 50, 2);
                neuralNetworkTick.setNeuralNetworkName("MIDI_tick NN");
            }
            else {
                neuralNetworkKey = Persistence.restoreNeuralNetwork(persistenceNameKey);
                neuralNetworkVelocity = Persistence.restoreNeuralNetwork(persistenceNameVelocity);
                neuralNetworkTick = Persistence.restoreNeuralNetwork(persistenceNameTick);
            }

            neuralNetworkKey.setTaskType(MetricsType.CLASSIFICATION);
            neuralNetworkVelocity.setTaskType(MetricsType.CLASSIFICATION);
            neuralNetworkTick.setTaskType(MetricsType.REGRESSION);

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

            String params = "randomOrder = false, randomStart = true, stepSize = 1, shuffleSamples = false, sampleSize = 32, numberOfIterations = 100";
            neuralNetworkKey.setTrainingData(new BasicSampler(data.get(0), data.get(1),params));
            neuralNetworkVelocity.setTrainingData(new BasicSampler(data.get(2), data.get(3),params));
            neuralNetworkTick.setTrainingData(new BasicSampler(data.get(4), data.get(5),params));

            int totalIterations = neuralNetworkKey.getTotalIterations();
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

                LinkedHashMap<Integer, MMatrix> resultKey = new LinkedHashMap<>();
                LinkedHashMap<Integer, MMatrix> resultVelocity = new LinkedHashMap<>();
                LinkedHashMap<Integer, MMatrix> resultTick = new LinkedHashMap<>();

                Matrix currentSampleKey = data.get(0).get(0).get(0);
                Matrix currentSampleVelocity = data.get(2).get(0).get(0);
                Matrix currentSampleTick = data.get(4).get(0).get(0);

                for (int sample = 0; sample < 1000; sample++) {
                    currentSampleKey = neuralNetworkKeyForPrediction.predict(currentSampleKey);
                    resultKey.put(sample, new MMatrix(currentSampleKey));
                    currentSampleVelocity = neuralNetworkVelocityForPrediction.predict(currentSampleVelocity);
                    resultVelocity.put(sample, new MMatrix(currentSampleVelocity));
                    currentSampleTick = neuralNetworkTickForPrediction.predict(currentSampleTick);
                    resultTick.put(sample, new MMatrix(currentSampleTick));
                }
                neuralNetworkKeyForPrediction.stop();
                neuralNetworkVelocityForPrediction.stop();
                neuralNetworkTickForPrediction.stop();

                System.out.println("Get MIDI sequence...");
                Sequence resultSequence = ReadMIDI.getSequence(resultKey, resultVelocity, resultTick, divisionType, resolution, scalingFactor);
                System.out.println("Play MIDI...");
                Sequencer sequencer = ReadMIDI.play(resultSequence, 30, false);

                neuralNetworkKey.waitToComplete();
                neuralNetworkVelocity.waitToComplete();
                neuralNetworkTick.waitToComplete();

                System.out.println("Play MIDI complete...");
                ReadMIDI.stopPlaying(sequencer);
                ReadMIDI.writeMIDI(resultSequence, path + "Result.mid");
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
     * Builds recurrent neural network (GRU) instance.
     *
     * @param inputSize input size of neural network (digits as one hot encoded in sequence).
     * @param outputSize output size of neural network (digits as one hot encoded in sequence).
     * @param hiddenSize hidden layer size of neural network.
     * @return neural network instance.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize, int hiddenSize, int lossType) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = " + inputSize);
        if (lossType > 1) {
            neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, "width = " + (hiddenSize / 2));
            neuralNetwork.addHiddenLayer(LayerType.GRU, "width = " + (hiddenSize / 2));
        }
        else neuralNetwork.addHiddenLayer(LayerType.GRU, "width = " + hiddenSize);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(lossType == 0 ? UnaryFunctionType.GUMBEL_SOFTMAX : lossType == 1 ? UnaryFunctionType.GUMBEL_SOFTMAX : UnaryFunctionType.RELU_SIN), "width = " + outputSize);
        neuralNetwork.addOutputLayer(lossType < 2 ? BinaryFunctionType.CROSS_ENTROPY : BinaryFunctionType.MEAN_SQUARED_ERROR);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.RADAM);
        return neuralNetwork;
    }

}
