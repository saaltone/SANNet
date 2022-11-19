/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package demo;

import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.network.NeuralNetwork;
import core.network.NeuralNetworkException;
import core.optimization.OptimizationType;
import core.preprocess.DataSplitter;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.sampling.BasicSampler;

import java.util.ArrayList;
import java.util.HashMap;
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

            int numberOfNeuralNetworks = 2;
            HashMap<Integer, HashMap<Integer, MMatrix>> data = getTestData();
            ArrayList<NeuralNetwork> neuralNetworks = new ArrayList<>();
            for (int index = 0; index < numberOfNeuralNetworks; index++) {
                NeuralNetwork neuralNetwork = buildNeuralNetwork(data.get(0).get(0).get(0).getRows(), data.get(1).get(0).get(0).getRows());
                neuralNetworks.add(neuralNetwork);
                initializeNeuralNetwork(neuralNetwork, index + 1, data, index == 0);
            }

            for (NeuralNetwork neuralNetwork : neuralNetworks) neuralNetwork.train(false, false);
            for (NeuralNetwork neuralNetwork : neuralNetworks) neuralNetwork.waitToComplete();

            Random random = new Random();
            for (int index = 0; index < 10; index++) {
                int input1 = random.nextInt(75);
                int input2 = random.nextInt(100 - input1);
                int result = input1 + input2;

                Matrix inputData = new DMatrix(2, 1);
                inputData.setValue(0, 0, (double)input1 / 100);
                inputData.setValue(1, 0, (double)input2 / 100);

                for (NeuralNetwork neuralNetwork : neuralNetworks) {
                    Matrix outputData = neuralNetwork.predict(inputData);
                    int predictedOutput = (int)(outputData.getValue(0, 0) * 100);
                    System.out.println(neuralNetwork.getNeuralNetworkName() + ": " + input1 + " + " + input2 + " = " + result + " (predicted result: " + predictedOutput + "), delta: " + (result - predictedOutput));
                }
            }

            for (NeuralNetwork neuralNetwork : neuralNetworks) neuralNetwork.stop();

        }
        catch (Exception exception) {
            exception.printStackTrace();
            System.exit(-1);
        }
    }

    private static void initializeNeuralNetwork(NeuralNetwork neuralNetwork, int id, HashMap<Integer, HashMap<Integer, MMatrix>> data, boolean print) throws NeuralNetworkException, MatrixException, DynamicParamException {
        neuralNetwork.setNeuralNetworkName("Neural Network " + id);
        neuralNetwork.setAsRegression();
        neuralNetwork.verboseTraining(10);
        neuralNetwork.setAutoValidate(5);
        neuralNetwork.verboseValidation();
//        neuralNetwork.setTrainingEarlyStopping(new EarlyStopping());
        neuralNetwork.start();
        if (print) {
            neuralNetwork.print();
            neuralNetwork.printExpressions();
            neuralNetwork.printGradients();
        }
        neuralNetwork.setTrainingData(new BasicSampler(data.get(0), data.get(1), "randomOrder = false, shuffleSamples = true, sampleSize = 100, numberOfIterations = 2500"));
        neuralNetwork.setValidationData(new BasicSampler(data.get(2), data.get(3), "randomOrder = false, shuffleSamples = true, sampleSize = " + data.get(2).size()));

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
        neuralNetwork.addHiddenLayer(LayerType.WEIGHT_NORMALIZATION);
        neuralNetwork.addHiddenLayer(LayerType.DENSE, "width = 20");
        neuralNetwork.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.RELU));
        neuralNetwork.addHiddenLayer(LayerType.CONNECTOR, "inputLayers = [0; 1], joinPreviousLayerInputs = true");
        neuralNetwork.addHiddenLayer(LayerType.DENSE, "width = " + outputSize);
        neuralNetwork.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.RELU));
        neuralNetwork.addOutputLayer(BinaryFunctionType.MEAN_SQUARED_ERROR);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.ADAM);
        return neuralNetwork;
    }

    /**
     * Creates training and test samples with split of 70% / 30%.
     * Inputs are two numbers between zero and 100 so that their sum is 100.
     * Output is result of summation of inputs.
     *
     * @return created training and testing samples.
     * @throws NeuralNetworkException throws exception if creation of samples fail.
     * @throws MatrixException throws exception if matrix is exceeding its depth or matrix is not defined.
     */
    private static HashMap<Integer, HashMap<Integer, MMatrix>> getTestData() throws NeuralNetworkException, MatrixException {
        HashMap<Integer, HashMap<Integer, MMatrix>> data = new HashMap<>();
        HashMap<Integer, MMatrix> input = new HashMap<>();
        HashMap<Integer, MMatrix> output = new HashMap<>();
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
