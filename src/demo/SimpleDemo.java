/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package demo;

import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.metrics.MetricsType;
import core.normalization.NormalizationType;
import core.optimization.*;
import core.preprocess.*;
import core.regularization.*;
import utils.*;
import core.*;
import utils.matrix.*;
import utils.sampling.BasicSampler;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Random;

/**
 * Creates two instances of neural network.
 * Neural network instances are learning to calculate two numbers together between zero and 100.
 * Instances are run concurrently in separate threads.
 *
 */
public class SimpleDemo {
    public static void main(String [] args) {

        try {
            HashMap<Integer, LinkedHashMap<Integer, Sample>> data = getTestData();
            NeuralNetwork neuralNetwork1 = buildNeuralNetwork(data.get(0).get(0).get(0).getRows(), data.get(1).get(0).get(0).getRows());
            neuralNetwork1.setNeuralNetworkName("Neural Network 1");
            neuralNetwork1.setTaskType(MetricsType.REGRESSION);
            neuralNetwork1.verboseTraining(10);
            neuralNetwork1.setAutoValidate(5);
            neuralNetwork1.verboseValidation();
            neuralNetwork1.setTrainingEarlyStopping(new EarlyStopping());
            neuralNetwork1.start();
            neuralNetwork1.setTrainingData(new BasicSampler(data.get(0), data.get(1), "randomOrder = false, shuffleSamples = true, sampleSize = 100"));
            neuralNetwork1.setValidationData(new BasicSampler(data.get(2), data.get(3), "randomOrder = false, shuffleSamples = true, sampleSize = " + data.get(2).size()));
            neuralNetwork1.setTrainingIterations(10000);

            NeuralNetwork neuralNetwork2 = buildNeuralNetwork(data.get(0).get(0).get(0).getRows(), data.get(1).get(0).get(0).getRows());
            neuralNetwork2.setNeuralNetworkName("Neural Network 2");
            neuralNetwork2.setTaskType(MetricsType.REGRESSION);
            neuralNetwork2.verboseTraining(10);
            neuralNetwork2.setAutoValidate(5);
            neuralNetwork2.verboseValidation();
            neuralNetwork2.setTrainingEarlyStopping(new EarlyStopping());
            neuralNetwork2.start();
            neuralNetwork2.setTrainingData(new BasicSampler(data.get(0), data.get(1), "randomOrder = false, shuffleSamples = true, sampleSize = 100"));
            neuralNetwork2.setValidationData(new BasicSampler(data.get(2), data.get(3), "randomOrder = false, shuffleSamples = true, sampleSize = " + data.get(2).size()));
            neuralNetwork2.setTrainingIterations(10000);

            neuralNetwork1.train(false, false);
            neuralNetwork2.train(false, false);

            neuralNetwork1.waitToComplete();
            neuralNetwork2.waitToComplete();

            neuralNetwork1.stop();
            neuralNetwork2.stop();
        }
        catch (Exception exception) {
            exception.printStackTrace();
            System.exit(-1);
        }
    }

    /**
     * Build neural network instance for regression.
     *
     * @param inputSize input layer size (in this case 2).
     * @param outputSize output layer size (in this case 1).
     * @return neural network instance.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     */
    private static NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize) throws MatrixException, DynamicParamException, NeuralNetworkException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = " + inputSize);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = 20");
        neuralNetwork.addOutputLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = " + outputSize);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.RADAM);
        neuralNetwork.addNormalizer(1, NormalizationType.WEIGHT_NORMALIZATION);
        neuralNetwork.setLossFunction(BinaryFunctionType.HUBER);
        return neuralNetwork;
    }

    /**
     * Creates training and test samples with split of 70% / 30%.
     * Inputs are two numbers between zero and 100 so that their sum is 100.
     * Output is result of summation of inputs.
     *
     * @return created training and testing samples.
     * @throws NeuralNetworkException throws exception if creation of samples fail.
     */
    private static HashMap<Integer, LinkedHashMap<Integer, Sample>> getTestData() throws NeuralNetworkException {
        HashMap<Integer, LinkedHashMap<Integer, Sample>> data = new HashMap<>();
        LinkedHashMap<Integer, Sample> input = new LinkedHashMap<>();
        LinkedHashMap<Integer, Sample> output = new LinkedHashMap<>();
        data.put(0, input);
        data.put(1, output);

        int sampleAmount = 10000;
        Random random = new Random();
        for (int i = 0; i < sampleAmount; i++) {
            int input1 = random.nextInt(75);
            int input2 = random.nextInt(100 - input1);
            int result = input1 + input2;

            Matrix inputData = new DMatrix(2, 1);
            Matrix outputData = new DMatrix(1, 1);
            inputData.setValue(0, 0, (double)input1 / 100);
            inputData.setValue(1, 0, (double)input2 / 100);
            input.put(i, new Sample(inputData));
            outputData.setValue(0, 0, (double)result / 100);
            output.put(i, new Sample(outputData));
        }
        data = DataSplitter.split(data, 0.3, false);
        return data;
    }
}
