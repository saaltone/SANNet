/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package demo;

import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.metrics.ClassificationMetric;
import core.network.EarlyStopping;
import core.network.NeuralNetwork;
import core.network.NeuralNetworkConfiguration;
import core.network.NeuralNetworkException;
import core.optimization.OptimizationType;
import core.preprocess.DataSplitter;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.sampling.BasicSampler;
import utils.sampling.Sequence;

import java.util.HashMap;
import java.util.Random;
import java.util.TreeMap;

/**
 * Demo with auto-encoder.
 * Encodes and decodes numbers from 0 to 10
 *
 */
public class AutoEncoder {

    /**
     * Default constructor for auto encoder demo.
     *
     */
    public AutoEncoder() {
    }

    /**
     * Main function for auto-encoder demo.
     *
     * @param args arguments
     */
    public static void main(String [] args) {

        try {

            int maxValue = 10;

            HashMap<Integer, HashMap<Integer, Matrix>> data = getTestData(maxValue);
            NeuralNetwork neuralNetwork = buildNeuralNetwork(data.get(0).get(0).getRows());
            initializeNeuralNetwork(neuralNetwork, data);

            neuralNetwork.train(true, true);

            int numberOfTests = 100;
            Random random = new Random();
            Sequence inputSequence = new Sequence();
            Sequence outputSequence = new Sequence();
            for (int index = 0; index < numberOfTests; index++) {
                int inputValue = random.nextInt(maxValue);

                Matrix inputData = new DMatrix(maxValue, 1, 1);
                inputSequence.put(index, inputData);
                inputData.setValue(inputValue, 0, 0, 1);

                Matrix outputData = neuralNetwork.predictMatrix(new TreeMap<>() {{ put(0, inputData); }}).get(0);
                outputSequence.put(index, outputData);
                int predictedOutput = outputData.argmax()[0];

                System.out.println(neuralNetwork.getNeuralNetworkName() + " Input: " + inputValue + ", Output: " + predictedOutput);
            }

            ClassificationMetric classificationMetric = new ClassificationMetric();
            classificationMetric.report(outputSequence, inputSequence);
            classificationMetric.printReport();

            neuralNetwork.stop();

            System.out.println("Removing last hidden layers for decoder part.");
            neuralNetwork.removeLastHiddenLayers(6);

            neuralNetwork.start();
            neuralNetwork.print();

            numberOfTests = maxValue;
            for (int index = 0; index < numberOfTests; index++) {
                int finalIndex = index;
                Matrix outputData = neuralNetwork.predictMatrix(new TreeMap<>() {{ put(0, DMatrix.getOneHotVector(maxValue, finalIndex)); }}).get(0);
                System.out.println(neuralNetwork.getNeuralNetworkName() + " Input: " + index + ", Output:");
                outputData.print();
            }

            neuralNetwork.stop();

        }
        catch (Exception exception) {
            exception.printStackTrace();
            System.exit(-1);
        }
    }

    /**
     * Initializes neural network.
     *
     * @param neuralNetwork neural network instance
     * @param data input and output data.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private static void initializeNeuralNetwork(NeuralNetwork neuralNetwork, HashMap<Integer, HashMap<Integer, Matrix>> data) throws NeuralNetworkException, MatrixException, DynamicParamException {
        neuralNetwork.setNeuralNetworkName("Neural Network " + 1);
        neuralNetwork.setAsClassification(true);
        neuralNetwork.verboseTraining(10);
        neuralNetwork.setAutoValidate(5);
        neuralNetwork.verboseValidation();
        neuralNetwork.setTrainingEarlyStopping(new TreeMap<>() {{ put(0, new EarlyStopping()); }});
        neuralNetwork.setShowTrainingMetrics(true);
        neuralNetwork.showConfusionMatrix(true);
        neuralNetwork.start();

        neuralNetwork.print();
        neuralNetwork.printExpressions();
        neuralNetwork.printGradients();

        neuralNetwork.setTrainingData(new BasicSampler(new HashMap<>() {{ put(0, data.get(0)); }}, new HashMap<>() {{ put(0, data.get(1)); }}, "randomOrder = false, shuffleSamples = true, sampleSize = 100, numberOfIterations = 10000"));
        neuralNetwork.setValidationData(new BasicSampler(new HashMap<>() {{ put(0, data.get(2)); }}, new HashMap<>() {{ put(0, data.get(3)); }}, "randomOrder = false, shuffleSamples = true, sampleSize = " + data.get(2).size()));
    }

    /**
     * Build neural network instance for regression.
     *
     * @param inputSize input layer size (in this case 10).
     * @return neural network instance.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static NeuralNetwork buildNeuralNetwork(int inputSize) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetworkConfiguration neuralNetworkConfiguration = new NeuralNetworkConfiguration();
        neuralNetworkConfiguration.addInputLayer("width = " + inputSize + ", height = 1, depth = 1");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.DENSE, "width = " + (inputSize - 3));
        neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.ELU));
        neuralNetworkConfiguration.addHiddenLayer(LayerType.DENSE, "width = " + (inputSize - 5));
        neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.ELU));
        neuralNetworkConfiguration.addHiddenLayer(LayerType.DENSE, "width = " + (inputSize - 5));
        neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.ELU));
        neuralNetworkConfiguration.addHiddenLayer(LayerType.DENSE, "width = " + (inputSize - 3));
        neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.ELU));
        neuralNetworkConfiguration.addHiddenLayer(LayerType.DENSE, "width = " + inputSize);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.SOFTMAX));
        neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.COS_SIM);
        neuralNetworkConfiguration.connectLayersSerially();

        NeuralNetwork neuralNetwork = new NeuralNetwork(neuralNetworkConfiguration);

        neuralNetwork.setOptimizer(OptimizationType.ADAM);

        return neuralNetwork;
    }

    /**
     * Creates training and test samples with split of 70% / 30%.
     * Inputs are numbers between zero and 10 as one-hot encoded.
     * Outputs are numbers between zero and 10 as one-hot encoded.
     *
     * @return created training and testing samples.
     * @throws NeuralNetworkException throws exception if creation of samples fail.
     * @throws MatrixException throws exception if matrix is exceeding its depth or matrix is not defined.
     */
    private static HashMap<Integer, HashMap<Integer, Matrix>> getTestData(int maxValue) throws NeuralNetworkException, MatrixException {
        HashMap<Integer, HashMap<Integer, Matrix>> data = new HashMap<>();
        HashMap<Integer, Matrix> input = new HashMap<>();
        HashMap<Integer, Matrix> output = new HashMap<>();
        data.put(0, input);
        data.put(1, output);

        Random random = new Random();
        for (int i = 0; i < 100 * maxValue; i++) {
            Matrix inputData = DMatrix.getOneHotVector(maxValue, random.nextInt(maxValue));
            input.put(i, inputData);
            output.put(i, inputData);
        }
        data = DataSplitter.split(data, 0.3, false);
        return data;
    }

}
