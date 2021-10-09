/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package demo;

import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.metrics.MetricsType;
import core.network.NeuralNetwork;
import core.network.NeuralNetworkException;
import core.normalization.NormalizationType;
import core.optimization.OptimizationType;
import core.preprocess.DataSplitter;
import core.regularization.EarlyStopping;
import utils.DynamicParamException;
import utils.matrix.*;
import utils.sampling.BasicSampler;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Random;

/**
 * Creates two instances of neural network.<br>
 * Neural network instances are learning to calculate two numbers together between zero and 100.<br>
 * Instances run concurrently in separate threads.<br>
 *
 */
public class SimpleDemo {

    /**
     * Main function for simple demo.
     *
     * @param args arguments
     */
    public static void main(String [] args) {

        try {
            HashMap<Integer, LinkedHashMap<Integer, MMatrix>> data = getTestData();
            NeuralNetwork neuralNetwork1 = buildNeuralNetwork(data.get(0).get(0).get(0).getRows(), data.get(1).get(0).get(0).getRows());
            neuralNetwork1.setNeuralNetworkName("Neural Network 1");
            neuralNetwork1.setTaskType(MetricsType.REGRESSION);
            neuralNetwork1.verboseTraining(10);
            neuralNetwork1.setAutoValidate(5);
            neuralNetwork1.verboseValidation();
            neuralNetwork1.setTrainingEarlyStopping(new EarlyStopping());
            neuralNetwork1.start();
            neuralNetwork1.print();
            neuralNetwork1.printExpressions();
            neuralNetwork1.printGradients();
            neuralNetwork1.setTrainingData(new BasicSampler(data.get(0), data.get(1), "randomOrder = false, shuffleSamples = true, sampleSize = 100, numberOfIterations = 10000000"));
            neuralNetwork1.setValidationData(new BasicSampler(data.get(2), data.get(3), "randomOrder = false, shuffleSamples = true, sampleSize = " + data.get(2).size()));

            NeuralNetwork neuralNetwork2 = buildNeuralNetwork(data.get(0).get(0).get(0).getRows(), data.get(1).get(0).get(0).getRows());
            neuralNetwork2.setNeuralNetworkName("Neural Network 2");
            neuralNetwork2.setTaskType(MetricsType.REGRESSION);
            neuralNetwork2.verboseTraining(10);
            neuralNetwork2.setAutoValidate(5);
            neuralNetwork2.verboseValidation();
            neuralNetwork2.setTrainingEarlyStopping(new EarlyStopping());
            neuralNetwork2.start();
            neuralNetwork2.setTrainingData(new BasicSampler(data.get(0), data.get(1), "randomOrder = false, shuffleSamples = true, sampleSize = 100, numberOfIterations = 10000000"));
            neuralNetwork2.setValidationData(new BasicSampler(data.get(2), data.get(3), "randomOrder = false, shuffleSamples = true, sampleSize = " + data.get(2).size()));

            neuralNetwork1.train(false, false);
            neuralNetwork2.train(false, false);

            neuralNetwork1.waitToComplete();
            neuralNetwork2.waitToComplete();

            Random random = new Random();
            for (int index = 0; index < 10; index++) {
                int input1 = random.nextInt(75);
                int input2 = random.nextInt(100 - input1);
                int result = input1 + input2;

                Matrix inputData = new DMatrix(2, 1);
                inputData.setValue(0, 0, (double)input1 / 100);
                inputData.setValue(1, 0, (double)input2 / 100);

                Matrix outputData = neuralNetwork1.predict(inputData);
                int predictedOutput = (int)(outputData.getValue(0, 0) * 100);
                System.out.println("Neural network 1: " + input1 + " + " + input2 + " = " + result + " (predicted result: " + predictedOutput + "), delta: " + (result - predictedOutput));

                outputData = neuralNetwork2.predict(inputData);
                predictedOutput = (int)(outputData.getValue(0, 0) * 100);
                System.out.println("Neural network 2: " + input1 + " + " + input2 + " = " + result + " (predicted result: " + predictedOutput + "), delta: " + (result - predictedOutput));
            }

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
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = " + inputSize);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = 20");
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = " + outputSize);
        neuralNetwork.addOutputLayer(BinaryFunctionType.HUBER);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.AMSGRAD);
        neuralNetwork.addNormalizer(1, NormalizationType.WEIGHT_NORMALIZATION);
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
    private static HashMap<Integer, LinkedHashMap<Integer, MMatrix>> getTestData() throws NeuralNetworkException {
        HashMap<Integer, LinkedHashMap<Integer, MMatrix>> data = new HashMap<>();
        LinkedHashMap<Integer, MMatrix> input = new LinkedHashMap<>();
        LinkedHashMap<Integer, MMatrix> output = new LinkedHashMap<>();
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
            input.put(i, new MMatrix(inputData));
            outputData.setValue(0, 0, (double)result / 100);
            output.put(i, new MMatrix(outputData));
        }
        data = DataSplitter.split(data, 0.3, false);
        return data;
    }
}
