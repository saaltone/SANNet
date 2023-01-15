/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package demo;

import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.network.NeuralNetwork;
import core.network.NeuralNetworkConfiguration;
import core.network.NeuralNetworkException;
import core.optimization.OptimizationType;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.sampling.BasicSampler;

import java.util.*;

/**
 * Demo for time series prediction. Uses sin function as input data.
 *
 */
public class TimeSeriesPrediction {

    /**
     * Main function for pseudo time series demo.
     *
     * @param args arguments
     */
    public static void main(String [] args) {

        try {
            int sampleAmount = 10000;
            int numberOfInputs = 5;
            double scalingFactor = 10;

            HashMap<Integer, HashMap<Integer, MMatrix>> trainingInputs = new HashMap<>();
            HashMap<Integer, HashMap<Integer, MMatrix>> trainingOutputs = new HashMap<>();
            HashMap<Integer, HashMap<Integer, MMatrix>> validationInputs = new HashMap<>();
            HashMap<Integer, HashMap<Integer, MMatrix>> validationOutputs = new HashMap<>();

            getTimeSeriesData(trainingInputs, trainingOutputs, validationInputs, validationOutputs, numberOfInputs, sampleAmount, scalingFactor);
            NeuralNetwork neuralNetwork = buildNeuralNetwork(trainingInputs.size(), trainingInputs.get(0).get(0).get(0).getRows(), trainingOutputs.get(0).get(0).get(0).getRows());
            initializeNeuralNetwork(neuralNetwork, trainingInputs, trainingOutputs, validationInputs, validationOutputs);

            neuralNetwork.resetDependencies(false);

            neuralNetwork.train(false, false);
            neuralNetwork.waitToComplete();

            double step = 1 / (double)sampleAmount;
            ArrayDeque<Double> currentYs = new ArrayDeque<>();
            for (int inputIndex = 0; inputIndex < numberOfInputs; inputIndex++) {
                currentYs.addLast(TimeSeriesPrediction.getTimeSeriesValue(step * (double)inputIndex, scalingFactor));
            }

            for (double inputIndex = 0; inputIndex < 1000; inputIndex++) {
                double nextT = step * inputIndex;
                double currentY = TimeSeriesPrediction.getTimeSeriesValue(nextT, scalingFactor);

                TreeMap<Integer, Matrix> inputs = new TreeMap<>();
                int inputOffset = 0;
                for (Double thisCurrentY : currentYs) {
                    System.out.print(thisCurrentY + " ");
                    Matrix inputData = new DMatrix(1, 1);
                    inputData.setValue(0, 0, thisCurrentY);
                    inputs.put(inputOffset++, inputData);
                }
                System.out.println();

                Matrix outputData = neuralNetwork.predictMatrix(inputs).get(0);
                double predictedY = outputData.getValue(0, 0);

                System.out.println(neuralNetwork.getNeuralNetworkName() + ": Current Y: " + currentY + ", Next T: " + nextT + ", Predicted Y " + predictedY + " (delta: " + (predictedY - currentY) + ")");

                currentYs.removeFirst();
                currentYs.add(predictedY);
            }

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
     * @param neuralNetwork neural network
     * @param trainingInputs training inputs
     * @param trainingOutputs training outputs
     * @param validationInputs validation inputs
     * @param validationOutputs validation outputs
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private static void initializeNeuralNetwork(NeuralNetwork neuralNetwork, HashMap<Integer, HashMap<Integer, MMatrix>> trainingInputs, HashMap<Integer, HashMap<Integer, MMatrix>> trainingOutputs, HashMap<Integer, HashMap<Integer, MMatrix>> validationInputs, HashMap<Integer, HashMap<Integer, MMatrix>> validationOutputs) throws NeuralNetworkException, MatrixException, DynamicParamException {
        neuralNetwork.setNeuralNetworkName("Neural Network");
        neuralNetwork.setAsRegression();
        neuralNetwork.verboseTraining(10);
        neuralNetwork.setAutoValidate(100);
        neuralNetwork.verboseValidation();
//        neuralNetwork.setTrainingEarlyStopping(new TreeMap<>() {{ put(0, new EarlyStopping("trainingStopThreshold = 100, validationStopThreshold = 100")); }});
        neuralNetwork.start();
        neuralNetwork.print();
        neuralNetwork.printExpressions();
        neuralNetwork.printGradients();
        neuralNetwork.setTrainingData(new BasicSampler(trainingInputs, trainingOutputs, "randomOrder = false, shuffleSamples = false, sampleSize = 100, numberOfIterations = 2500"));
        neuralNetwork.setValidationData(new BasicSampler(validationInputs, validationOutputs, "randomOrder = false, shuffleSamples = false, sampleSize = " + validationInputs.get(0).size()));
    }

    /**
     * Build neural network instance for regression.
     *
     * @param numberOfInputs number of inputs.
     * @param inputSize input layer size.
     * @param outputSize output layer size.
     * @return neural network instance.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static NeuralNetwork buildNeuralNetwork(int numberOfInputs, int inputSize, int outputSize) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetworkConfiguration neuralNetworkConfiguration = new NeuralNetworkConfiguration();
        int[] hiddenLayerIndices = new int[numberOfInputs];
        for (int i = 0; i < numberOfInputs; i++) {
            int inputLayerIndex = neuralNetworkConfiguration.addInputLayer("width = " + inputSize);
            int hiddenLayerIndex1 = neuralNetworkConfiguration.addHiddenLayer(LayerType.GRU, "width = 20");
            int hiddenLayerIndex2 = neuralNetworkConfiguration.addHiddenLayer(LayerType.GRU, "width = 20, reversedInput = true");
            neuralNetworkConfiguration.connectLayers(inputLayerIndex, hiddenLayerIndex1);
            neuralNetworkConfiguration.connectLayers(inputLayerIndex, hiddenLayerIndex2);
            hiddenLayerIndices[i] = neuralNetworkConfiguration.addHiddenLayer(LayerType.JOIN);
            neuralNetworkConfiguration.connectLayers(hiddenLayerIndex1, hiddenLayerIndices[i]);
            neuralNetworkConfiguration.connectLayers(hiddenLayerIndex2, hiddenLayerIndices[i]);
        }
        int joinLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.ATTENTION);
        for (int i = 0; i < numberOfInputs; i++) neuralNetworkConfiguration.connectLayers(hiddenLayerIndices[i], joinLayerIndex);
        int hiddenLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.TANH), "width = " + outputSize);
        neuralNetworkConfiguration.connectLayers(joinLayerIndex, hiddenLayerIndex);
        int outputLayerIndex = neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.MEAN_SQUARED_ERROR);
        neuralNetworkConfiguration.connectLayers(hiddenLayerIndex, outputLayerIndex);

        NeuralNetwork neuralNetwork = new NeuralNetwork(neuralNetworkConfiguration);

        neuralNetwork.setOptimizer(OptimizationType.ADAM);

        return neuralNetwork;
    }

    /**
     * Returns time series data
     *
     * @param trainingInputs training inputs
     * @param trainingOutputs training outputs
     * @param validationInputs validation inputs
     * @param validationOutputs validation outputs
     * @param numberOfInputs number of input
     * @param sampleAmount amount of samples
     * @param scalingFactor scaling factor
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private static void getTimeSeriesData(HashMap<Integer, HashMap<Integer, MMatrix>> trainingInputs, HashMap<Integer, HashMap<Integer, MMatrix>> trainingOutputs, HashMap<Integer, HashMap<Integer, MMatrix>> validationInputs, HashMap<Integer, HashMap<Integer, MMatrix>> validationOutputs, int numberOfInputs, int sampleAmount, double scalingFactor) throws MatrixException {
        ArrayList<Double> values = getTimeSeriesValues(sampleAmount, scalingFactor);

        ArrayList<MMatrix> dataSet = new ArrayList<>();
        for (int i = 0; i < sampleAmount; i++) {
            Matrix inputData = new DMatrix(1, 1);
            inputData.setValue(0, 0, values.get(i));
            dataSet.add(new MMatrix(inputData));
        }

        for (int i = 0; i < numberOfInputs; i++) {
            trainingInputs.put(i, new HashMap<>());
            validationInputs.put(i, new HashMap<>());
        }
        trainingOutputs.put(0, new HashMap<>());
        validationOutputs.put(0, new HashMap<>());

        double validationShare = (1 - 0.1);
        int validationOffset = (int)((double)sampleAmount * validationShare);

        int trainingIndex = 0;
        int validationIndex = 0;
        for (int i = 0; i < sampleAmount - numberOfInputs - 1; i++) {
            for (int j = 0; j < numberOfInputs + 1; j++) {
                if (i < validationOffset) {
                    if (j < numberOfInputs) trainingInputs.get(j).put(trainingIndex, dataSet.get(i + j));
                    else trainingOutputs.get(0).put(trainingIndex++, dataSet.get(i + j));
                }
                else {
                    if (j < numberOfInputs) validationInputs.get(j).put(validationIndex, dataSet.get(i + j));
                    else validationOutputs.get(0).put(validationIndex++, dataSet.get(i + j));
                }
            }
        }

    }

    /**
     * Returns time series values.
     *
     * @param amount amount of values
     * @param scalingFactor scaling factor
     * @return time series values.
     */
    private static ArrayList<Double> getTimeSeriesValues(int amount, double scalingFactor) {
        double stepSize = 1 / (double)amount;
        ArrayList<Double> values = new ArrayList<>();
        for (double t = 0; t <= 1; t += stepSize) {
            values.add(getTimeSeriesValue(t, scalingFactor));
        }
        return values;
    }

    /**
     * Returns time series value
     *
     * @param t time
     * @param scalingFactor scaling factor
     * @return time series value
     */
    private static double getTimeSeriesValue(double t, double scalingFactor) {
        return Math.sin(2 * Math.PI * t * scalingFactor);
    }

}

