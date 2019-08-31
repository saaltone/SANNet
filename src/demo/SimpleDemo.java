/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package demo;

import core.activation.ActivationFunction;
import core.activation.ActivationFunctionType;
import core.layer.LayerType;
import core.loss.LossFunctionType;
import core.metrics.MetricsType;
import core.optimization.*;
import core.preprocess.*;
import core.regularization.*;
import utils.*;
import core.*;

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
            HashMap<Integer, LinkedHashMap<Integer, Matrix>> data = getTestData();
            NeuralNetwork neuralNetwork1 = buildNeuralNetwork(data.get(0).get(0).getRows(), data.get(1).get(0).getRows());
            neuralNetwork1.setNeuralNetworkName("Neural Network 1");
            neuralNetwork1.setTaskType(MetricsType.REGRESSION);
            neuralNetwork1.verboseTraining(10);
            neuralNetwork1.setAutoValidate(5);
            neuralNetwork1.verboseValidation();
            neuralNetwork1.setTrainingEarlyStopping(new EarlyStopping());
            neuralNetwork1.start();
            neuralNetwork1.setTrainingData(data.get(0), data.get(1));
            neuralNetwork1.setTrainingSampling(100, false, true);
            neuralNetwork1.setValidationData(data.get(2), data.get(3));
            neuralNetwork1.setValidationSampling(data.get(2).size(), true, false);
            neuralNetwork1.setTrainingIterations(10000);

            NeuralNetwork neuralNetwork2 = buildNeuralNetwork(data.get(0).get(0).getRows(), data.get(1).get(0).getRows());
            neuralNetwork2.setNeuralNetworkName("Neural Network 2");
            neuralNetwork2.setTaskType(MetricsType.REGRESSION);
            neuralNetwork2.verboseTraining(10);
            neuralNetwork2.setAutoValidate(5);
            neuralNetwork2.verboseValidation();
            neuralNetwork2.setTrainingEarlyStopping(new EarlyStopping());
            neuralNetwork2.start();
            neuralNetwork2.setTrainingData(data.get(0), data.get(1));
            neuralNetwork2.setTrainingSampling(100, false, true);
            neuralNetwork2.setValidationData(data.get(2), data.get(3));
            neuralNetwork2.setValidationSampling(data.get(2).size(), true, false);
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
    private static NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize) throws DynamicParamException, NeuralNetworkException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = " + inputSize);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(ActivationFunctionType.ELU), "width = 20");
        neuralNetwork.addOutputLayer(LayerType.FEEDFORWARD, new ActivationFunction(ActivationFunctionType.ELU), "width = " + outputSize);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.AMSGRAD);
        neuralNetwork.setLossFunction(LossFunctionType.HUBER);
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
    private static HashMap<Integer, LinkedHashMap<Integer, Matrix>> getTestData() throws NeuralNetworkException {
        HashMap<Integer, LinkedHashMap<Integer, Matrix>> data = new HashMap<>();
        LinkedHashMap<Integer, Matrix> input = new LinkedHashMap<>();
        LinkedHashMap<Integer, Matrix> output = new LinkedHashMap<>();
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
            input.put(i, inputData);
            outputData.setValue(0, 0, (double)result / 100);
            output.put(i, outputData);
        }
        data = DataSplitter.split(data, 0.3, false);
        return data;
    }
}
