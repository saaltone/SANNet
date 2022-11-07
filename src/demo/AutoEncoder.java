/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package demo;

import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.metrics.ClassificationMetric;
import core.network.EarlyStopping;
import core.network.NeuralNetwork;
import core.network.NeuralNetworkException;
import core.optimization.OptimizationType;
import core.preprocess.DataSplitter;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.sampling.BasicSampler;
import utils.sampling.Sequence;

import java.util.HashMap;
import java.util.Random;

/**
 * Demo with auto-encoder.
 * Encodes and decodes numbers from 0 to 10
 *
 */
public class AutoEncoder {

    /**
     * Main function for auto-encoder demo.
     *
     * @param args arguments
     */
    public static void main(String [] args) {

        try {

            int maxValue = 10;

            HashMap<Integer, HashMap<Integer, MMatrix>> data = getTestData(maxValue);
            NeuralNetwork neuralNetwork = buildNeuralNetwork(data.get(0).get(0).get(0).getRows(), data.get(1).get(0).get(0).getRows());
            initializeNeuralNetwork(neuralNetwork, data);

            neuralNetwork.train(true, false);
            neuralNetwork.waitToComplete();

            int numberOfTests = 100;
            Random random = new Random();
            Sequence inputSequence = new Sequence();
            Sequence outputSequence = new Sequence();
            for (int index = 0; index < numberOfTests; index++) {
                int inputValue = random.nextInt(maxValue);

                Matrix inputData = new DMatrix(maxValue, 1);
                inputSequence.put(index, new MMatrix(inputData));
                inputData.setValue(inputValue, 0, 1);

                Matrix outputData = neuralNetwork.predict(inputData);
                outputSequence.put(index, new MMatrix(outputData));
                int predictedOutput = outputData.argmax()[0];
                System.out.println(neuralNetwork.getNeuralNetworkName() + " Input: " + inputValue + ", Output: " + predictedOutput);
            }

            ClassificationMetric classificationMetric = new ClassificationMetric();
            classificationMetric.report(outputSequence, inputSequence);
            classificationMetric.printReport();

            neuralNetwork.stop();

        }
        catch (Exception exception) {
            exception.printStackTrace();
            System.exit(-1);
        }
    }

    /**
     * Initializes neural network
     *
     * @param neuralNetwork neural network.
     * @param data input and output data.
     * @throws NeuralNetworkException throws neural network exception is initialization of neural network fails.
     * @throws MatrixException throws matrix exception is matrix operation fails.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     */
    private static void initializeNeuralNetwork(NeuralNetwork neuralNetwork, HashMap<Integer, HashMap<Integer, MMatrix>> data) throws NeuralNetworkException, MatrixException, DynamicParamException {
        neuralNetwork.setNeuralNetworkName("Neural Network " + 1);
        neuralNetwork.setAsClassification();
        neuralNetwork.verboseTraining(10);
        neuralNetwork.setAutoValidate(5);
        neuralNetwork.verboseValidation();
        neuralNetwork.setTrainingEarlyStopping(new EarlyStopping("trainingStopThreshold = 25, validationStopThreshold = 25"));
        neuralNetwork.start();

        neuralNetwork.print();
        neuralNetwork.printExpressions();
        neuralNetwork.printGradients();

        neuralNetwork.setTrainingData(new BasicSampler(data.get(0), data.get(1), "randomOrder = false, shuffleSamples = true, sampleSize = 100, numberOfIterations = 10000"));
        neuralNetwork.setValidationData(new BasicSampler(data.get(2), data.get(3), "randomOrder = false, shuffleSamples = true, sampleSize = " + data.get(2).size()));
    }

    /**
     * Build neural network instance for regression.
     *
     * @param inputSize input layer size (in this case 10).
     * @param outputSize output layer size (in this case 10).
     * @return neural network instance.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = " + inputSize);
        neuralNetwork.addHiddenLayer(LayerType.DENSE, "width = " + (inputSize - 3));
        neuralNetwork.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.ELU));
        neuralNetwork.addHiddenLayer(LayerType.DENSE, "width = " + (inputSize - 5));
        neuralNetwork.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.ELU));
        neuralNetwork.addHiddenLayer(LayerType.DENSE, "width = " + (inputSize - 5));
        neuralNetwork.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.ELU));
        neuralNetwork.addHiddenLayer(LayerType.DENSE, "width = " + (inputSize - 3));
        neuralNetwork.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.ELU));
        neuralNetwork.addHiddenLayer(LayerType.DENSE, "width = " + inputSize);
        neuralNetwork.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.SOFTMAX));
        neuralNetwork.addOutputLayer(BinaryFunctionType.COS_SIM);

        neuralNetwork.build();

        neuralNetwork.setOptimizer(OptimizationType.ADAM);

        return neuralNetwork;
    }

    /**
     * Creates training and test samples with split of 70% / 30%.
     * Inputs are one hot encoded numbers between 1 and 10.
     * Outputs are one hot encoded numbers between 1 and 10.
     *
     * @return created training and testing samples.
     * @throws NeuralNetworkException throws exception if creation of samples fail.
     * @throws MatrixException throws exception if matrix is exceeding its depth or matrix is not defined.
     */
    private static HashMap<Integer, HashMap<Integer, MMatrix>> getTestData(int maxValue) throws NeuralNetworkException, MatrixException {
        HashMap<Integer, HashMap<Integer, MMatrix>> data = new HashMap<>();
        HashMap<Integer, MMatrix> input = new HashMap<>();
        HashMap<Integer, MMatrix> output = new HashMap<>();
        data.put(0, input);
        data.put(1, output);

        double scaleFactor = 2 / (double)maxValue;
        Random random = new Random();
        for (int i = 0; i < 100 * maxValue; i++) {
            int inputValue = random.nextInt(maxValue);

            Matrix inputData = new DMatrix(maxValue, 1);
            inputData.setValue(inputValue, 0, 1);
            MMatrix inputs = new MMatrix(inputData);
            input.put(i, inputs);
            output.put(i, inputs);
        }
        data = DataSplitter.split(data, 0.3, false);
        return data;
    }

}
