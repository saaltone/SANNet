/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package demo;

import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.network.EarlyStopping;
import core.network.NeuralNetwork;
import core.network.NeuralNetworkConfiguration;
import core.network.NeuralNetworkException;
import core.optimization.OptimizationType;
import core.preprocess.DataSplitter;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.sampling.BasicSampler;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeMap;

/**
 * Creates two instances of neural network.<br>
 * Neural network instances are learning to calculate two numbers together between zero and 100.<br>
 * Instances run concurrently in separate threads.<br>
 *
 */
public class SimpleDemo {

    /**
     * Default constructor for simple demo.
     *
     */
    public SimpleDemo() {
    }

    /**
     * Main function for simple demo.
     *
     * @param args arguments
     */
    public static void main(String [] args) {

        try {

            int numberOfNeuralNetworks = 2;
            HashMap<Integer, HashMap<Integer, Matrix>> data = getTestData();
            ArrayList<NeuralNetwork> neuralNetworks = new ArrayList<>();
            for (int index = 0; index < numberOfNeuralNetworks; index++) {
                NeuralNetwork neuralNetwork = buildNeuralNetwork(data.get(0).get(0).getRows(), data.get(1).get(0).getRows());
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

                Matrix inputData = new DMatrix(2, 1, 1);
                inputData.setValue(0, 0, 0, (double)input1 / 100);
                inputData.setValue(1, 0, 0, (double)input2 / 100);

                for (NeuralNetwork neuralNetwork : neuralNetworks) {
                    Matrix outputData = neuralNetwork.predictMatrix(new TreeMap<>() {{ put(0, inputData); }}).get(0);
                    int predictedOutput = (int)(outputData.getValue(0, 0, 0) * 100);
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

    private static void initializeNeuralNetwork(NeuralNetwork neuralNetwork, int id, HashMap<Integer, HashMap<Integer, Matrix>> data, boolean print) throws NeuralNetworkException, MatrixException, DynamicParamException {
        neuralNetwork.setNeuralNetworkName("Neural Network " + id);
        neuralNetwork.setAsRegression(true);
        neuralNetwork.verboseTraining(10);
        neuralNetwork.setAutoValidate(5);
        neuralNetwork.verboseValidation();
        neuralNetwork.setTrainingEarlyStopping(new TreeMap<>() {{ put(0, new EarlyStopping()); }});
        neuralNetwork.setShowTrainingMetrics(true);
        neuralNetwork.start();
        if (print) {
            neuralNetwork.print();
            neuralNetwork.printExpressions();
            neuralNetwork.printGradients();
        }
        neuralNetwork.setTrainingData(new BasicSampler(new HashMap<>() {{ put(0, data.get(0)); }}, new HashMap<>() {{ put(0, data.get(1)); }}, "randomOrder = false, shuffleSamples = true, sampleSize = 100, numberOfIterations = 2500"));
        neuralNetwork.setValidationData(new BasicSampler(new HashMap<>() {{ put(0, data.get(2)); }}, new HashMap<>() {{ put(0, data.get(3)); }}, "randomOrder = false, shuffleSamples = true, sampleSize = " + data.get(2).size()));

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
        NeuralNetworkConfiguration neuralNetworkConfiguration = new NeuralNetworkConfiguration();
        int inputLayerIndex = neuralNetworkConfiguration.addInputLayer("width = " + inputSize + ", height = 1, depth = 1");
        int hiddenLayerIndex1 = neuralNetworkConfiguration.addHiddenLayer(LayerType.WEIGHT_NORMALIZATION);
        int hiddenLayerIndex2 = neuralNetworkConfiguration.addHiddenLayer(LayerType.DENSE, "width = 20");
        int hiddenLayerIndex3 = neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.ELU));
        neuralNetworkConfiguration.addHiddenLayer(LayerType.GRADIENT_CLIPPING);
        int hiddenLayerIndex4 = neuralNetworkConfiguration.addHiddenLayer(LayerType.CONNECT);
        int hiddenLayerIndex5 = neuralNetworkConfiguration.addHiddenLayer(LayerType.DENSE, "width = " + outputSize);
        int hiddenLayerIndex6 = neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.RELU));
        int outputLayerIndex = neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.MEAN_SQUARED_ERROR);
        neuralNetworkConfiguration.connectLayersSerially();
        neuralNetworkConfiguration.connectLayers(hiddenLayerIndex1, hiddenLayerIndex4);

        NeuralNetwork neuralNetwork = new NeuralNetwork(neuralNetworkConfiguration);

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
     */
    private static HashMap<Integer, HashMap<Integer, Matrix>> getTestData() throws NeuralNetworkException {
        HashMap<Integer, HashMap<Integer, Matrix>> data = new HashMap<>();
        HashMap<Integer, Matrix> input = new HashMap<>();
        HashMap<Integer, Matrix> output = new HashMap<>();
        data.put(0, input);
        data.put(1, output);

        int sampleAmount = 10000;
        Random random = new Random();
        for (int i = 0; i < sampleAmount; i++) {
            int input1 = random.nextInt(75);
            int input2 = random.nextInt(100 - input1);
            int result = input1 + input2;

            Matrix inputData = new DMatrix(2, 1, 1);
            Matrix outputData = new DMatrix(1, 1, 1);
            inputData.setValue(0, 0, 0, (double)input1 / 100);
            inputData.setValue(1, 0, 0, (double)input2 / 100);
            input.put(i, inputData);
            outputData.setValue(0, 0, 0, (double)result / 100);
            output.put(i, outputData);
        }
        data = DataSplitter.split(data, 0.3, false);
        return data;
    }

}
