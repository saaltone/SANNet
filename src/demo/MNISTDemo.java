/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package demo;

import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.metrics.ClassificationMetric;
import core.network.*;
import core.optimization.OptimizationType;
import core.preprocess.ReadCSVFile;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.sampling.BasicSampler;
import utils.sampling.Sequence;

import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeMap;

/**
 * Implements learning and prediction of MNIST digits by using Convolutional Neural Network (CNN).<br>
 * Implementation reads training and test samples from CSV file and then executes learning process.<br>
 *
 */
public class MNISTDemo {

    /**
     * Default constructor for MNIST demo.
     *
     */
    public MNISTDemo() {
    }

    /**
     * Main function for demo.<br>
     * Function reads samples from files, build CNN and executed training, validation and testing.<br>
     *
     * @param args input arguments (not used).
     */
    public static void main(String [] args) {

        NeuralNetwork neuralNetwork;
        try {
            HashMap<Integer, HashMap<Integer, Matrix>> trainMNIST = getMNISTData(true);
            HashMap<Integer, HashMap<Integer, Matrix>> testMNIST = getMNISTData(false);

            neuralNetwork = buildNeuralNetwork(trainMNIST.get(0).get(0).getRows(), trainMNIST.get(1).get(0).getRows());

            String persistenceName = "<PATH>/MNIST_NN";
//            neuralNetwork = Persistence.restoreNeuralNetwork(persistenceName);

            Persistence persistence = new Persistence(true, 100, neuralNetwork, persistenceName, true);
            neuralNetwork.setPersistence(persistence);

            neuralNetwork.setAsClassification(true);
            neuralNetwork.verboseTraining(10);
            neuralNetwork.setAutoValidate(10);
            neuralNetwork.verboseValidation();
            neuralNetwork.setTrainingEarlyStopping(new TreeMap<>() {{ put(0, new EarlyStopping()); }});
            neuralNetwork.setShowTrainingMetrics(true);
            neuralNetwork.showConfusionMatrix(true);

            neuralNetwork.start();

            neuralNetwork.setTrainingData(new BasicSampler(new HashMap<>() {{ put(0, trainMNIST.get(0)); }}, new HashMap<>() {{ put(0, trainMNIST.get(1)); }},"perEpoch = true, randomOrder = true, shuffleSamples = true, sampleSize = 16, numberOfIterations = 5625"));
            neuralNetwork.setValidationData(new BasicSampler(new HashMap<>() {{ put(0, testMNIST.get(0)); }}, new HashMap<>() {{ put(0, testMNIST.get(1)); }}, "randomOrder = true, shuffleSamples = true, sampleSize = 20"));

            neuralNetwork.print();
            neuralNetwork.printExpressions();
            neuralNetwork.printGradients();

            System.out.println("Training...");
            neuralNetwork.train();

            System.out.println("Predicting...");

            ClassificationMetric predictionMetric = new ClassificationMetric();
            for (int index = 0; index < 100; index++) {
                Sequence input = new Sequence();
                Sequence output = new Sequence();
                for (int index1 = 0; index1 < 100; index1++) {
                    input.put(index1, testMNIST.get(0).get(index * 100 + index1));
                    output.put(index1, testMNIST.get(1).get(index * 100 + index1));
                }
                Sequence predict = neuralNetwork.predict(new TreeMap<>() {{ put(0, input); }}).get(0);
                if (index == 0) {
                    System.out.println("Printing out first 100 predictions...");
                    for (int index1 = 0; index1 < 100; index1++) {
                        int[] trueIndex = output.get(index1).argmax();
                        int[] predictIndex = predict.get(index1).argmax();
                        System.out.println("True label: " + trueIndex[0] + ", Predicted label: " + predictIndex[0]);
                    }
                }
                predictionMetric.report(predict, output);
            }
            predictionMetric.printReport();

            Persistence.saveNeuralNetwork(persistenceName, neuralNetwork);

            neuralNetwork.stop();
        }
        catch (Exception exception) {
            exception.printStackTrace();
            System.exit(-1);
        }
    }

    /**
     * Function that builds Convolutional Neural Network (CNN).<br>
     * MNIST images (size 28x28) are used as inputs.<br>
     * Outputs are 10 discrete outputs where each output represents single digit.<br>
     *
     * @param inputSize input size of convolutional neural network.
     * @param outputSize output size of convolutional neural network.
     * @return CNN instance.
     * @throws DynamicParamException throws exception is setting of parameters fails.
     * @throws NeuralNetworkException throws exception if creation of CNN fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetworkConfiguration neuralNetworkConfiguration = new NeuralNetworkConfiguration();
        int inputLayerIndex = neuralNetworkConfiguration.addInputLayer("width = " + inputSize + ", height = " + inputSize + ", depth = 1");
        int convLayerIndex0 = neuralNetworkConfiguration.addHiddenLayer(LayerType.SINGLE_DW_CROSSCORRELATION, Initialization.UNIFORM_XAVIER_CONV, "filterSize = 3, stride = 1");
        neuralNetworkConfiguration.connectLayers(inputLayerIndex, convLayerIndex0);
        int[] convLayerIndices1 = new int[24];
        for (int convLayerIndex1 = 0; convLayerIndex1 < convLayerIndices1.length; convLayerIndex1++) {
            int convLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.SINGLE_PW_CROSSCORRELATION, Initialization.UNIFORM_XAVIER_CONV);
            neuralNetworkConfiguration.connectLayers(convLayerIndex0, convLayerIndex);
            int batchNormLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.BATCH_NORMALIZATION);
            neuralNetworkConfiguration.connectLayers(convLayerIndex, batchNormLayerIndex);
            int activationLayer = neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.RELU));
            neuralNetworkConfiguration.connectLayers(batchNormLayerIndex, activationLayer);
            int poolingLayer = neuralNetworkConfiguration.addHiddenLayer(LayerType.SINGLE_AVERAGE_POOLING, "filterSize = 2, stride = 1");
            neuralNetworkConfiguration.connectLayers(activationLayer, poolingLayer);
            convLayerIndices1[convLayerIndex1] = neuralNetworkConfiguration.addHiddenLayer(LayerType.SINGLE_DW_CROSSCORRELATION, Initialization.UNIFORM_XAVIER_CONV, "filterSize = 3, stride = 1");
            neuralNetworkConfiguration.connectLayers(poolingLayer, convLayerIndices1[convLayerIndex1]);
        }
        int[] convLayerIndices2 = new int[24];
        for (int convLayerIndex2 = 0; convLayerIndex2 < convLayerIndices2.length; convLayerIndex2++) {
            int convLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.SINGLE_PW_CROSSCORRELATION, Initialization.UNIFORM_XAVIER_CONV);
            for (int layerIndex0 : convLayerIndices1) neuralNetworkConfiguration.connectLayers(layerIndex0, convLayerIndex);
            int batchNormLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.BATCH_NORMALIZATION);
            neuralNetworkConfiguration.connectLayers(convLayerIndex, batchNormLayerIndex);
            int activationLayer = neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.RELU));
            neuralNetworkConfiguration.connectLayers(batchNormLayerIndex, activationLayer);
            int poolingLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.SINGLE_AVERAGE_POOLING, "filterSize = 2, stride = 1");
            neuralNetworkConfiguration.connectLayers(activationLayer,poolingLayerIndex);
            int flattenLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.FLATTEN);
            neuralNetworkConfiguration.connectLayers(poolingLayerIndex,flattenLayerIndex);
            int dense0LayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.DENSE, "width = 100");
            neuralNetworkConfiguration.connectLayers(flattenLayerIndex, dense0LayerIndex);
            convLayerIndices2[convLayerIndex2] = neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.RELU));
            neuralNetworkConfiguration.connectLayers(dense0LayerIndex, convLayerIndices2[convLayerIndex2]);
        }

        int connectLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.CONNECT, "width = " + outputSize * 10);
        for (int convLayerIndex2 : convLayerIndices2) neuralNetworkConfiguration.connectLayers(convLayerIndex2, connectLayerIndex);

        int dense1LayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.DENSE, "width = " + outputSize);
        neuralNetworkConfiguration.connectLayers(connectLayerIndex, dense1LayerIndex);
        int activation1LayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.SOFTMAX));
        neuralNetworkConfiguration.connectLayers(dense1LayerIndex, activation1LayerIndex);
        int outputLayerIndex = neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
        neuralNetworkConfiguration.connectLayers(activation1LayerIndex, outputLayerIndex);

        NeuralNetwork neuralNetwork = new NeuralNetwork(neuralNetworkConfiguration);

        neuralNetwork.setOptimizer(OptimizationType.ADADELTA);
        return neuralNetwork;

/*        NeuralNetworkConfiguration neuralNetworkConfiguration = new NeuralNetworkConfiguration();
        int inputLayerIndex = neuralNetworkConfiguration.addInputLayer("width = " + inputSize + ", height = " + inputSize + ", depth = 1");
        int[] convLayerIndices1 = new int[12];
        for (int layerIndex = 0; layerIndex < convLayerIndices1.length; layerIndex++) {
            int convLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.SINGLE_CROSSCORRELATION, Initialization.UNIFORM_XAVIER_CONV, "filterSize = 3, stride = 1");
            neuralNetworkConfiguration.connectLayers(inputLayerIndex, convLayerIndex);
            int activationLayer = neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.RELU));
            neuralNetworkConfiguration.connectLayers(convLayerIndex, activationLayer);
            convLayerIndices1[layerIndex] = neuralNetworkConfiguration.addHiddenLayer(LayerType.SINGLE_AVERAGE_POOLING, "filterSize = 2, stride = 1");
            neuralNetworkConfiguration.connectLayers(activationLayer, convLayerIndices1[layerIndex]);
        }
        int[] convLayerIndices2 = new int[24];
        for (int layerIndex = 0; layerIndex < convLayerIndices2.length; layerIndex++) {
            int convLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.SINGLE_CROSSCORRELATION, Initialization.UNIFORM_XAVIER_CONV, "filterSize = 3, stride = 1");
            for (int layerIndex0 : convLayerIndices1) neuralNetworkConfiguration.connectLayers(layerIndex0, convLayerIndex);
            int activationLayer = neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.RELU));
            neuralNetworkConfiguration.connectLayers(convLayerIndex, activationLayer);
            int poolingLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.SINGLE_AVERAGE_POOLING, "filterSize = 2, stride = 1");
            neuralNetworkConfiguration.connectLayers(activationLayer,poolingLayerIndex);
            convLayerIndices2[layerIndex] = neuralNetworkConfiguration.addHiddenLayer(LayerType.FLATTEN);
            neuralNetworkConfiguration.connectLayers(poolingLayerIndex, convLayerIndices2[layerIndex]);
        }
        int joinLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.JOIN);
        for (int layerIndex1 : convLayerIndices2) neuralNetworkConfiguration.connectLayers(layerIndex1, joinLayerIndex);

        int batchNormLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.BATCH_NORMALIZATION);
        neuralNetworkConfiguration.connectLayers(joinLayerIndex, batchNormLayerIndex);
        int dense0LayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.DENSE, "width = 100");
        neuralNetworkConfiguration.connectLayers(batchNormLayerIndex, dense0LayerIndex);
        int activation0LayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.RELU));
        neuralNetworkConfiguration.connectLayers(dense0LayerIndex, activation0LayerIndex);
        int dense1LayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.DENSE, "width = " + outputSize);
        neuralNetworkConfiguration.connectLayers(activation0LayerIndex, dense1LayerIndex);
        int activation1LayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.SOFTMAX));
        neuralNetworkConfiguration.connectLayers(dense1LayerIndex, activation1LayerIndex);
        int outputLayerIndex = neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
        neuralNetworkConfiguration.connectLayers(activation1LayerIndex, outputLayerIndex);

        NeuralNetwork neuralNetwork = new NeuralNetwork(neuralNetworkConfiguration);

        neuralNetwork.setOptimizer(OptimizationType.ADADELTA);
        return neuralNetwork; */

/*        NeuralNetworkConfiguration neuralNetworkConfiguration = new NeuralNetworkConfiguration();
        neuralNetworkConfiguration.addInputLayer("width = " + inputSize + ", height = " + inputSize);
//        neuralNetworkConfiguration.addHiddenLayer(LayerType.CROSSCORRELATION, Initialization.UNIFORM_XAVIER_CONV, "filters = 12, filterSize = 3, stride = 1");
//        neuralNetworkConfiguration.addHiddenLayer(LayerType.WINOGRAD_CONVOLUTION, Initialization.UNIFORM_XAVIER_CONV, "filters = 12");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.DSCROSSCORRELATION, Initialization.UNIFORM_XAVIER_CONV, "filters = 12, filterSize = 3, stride = 1");
//        neuralNetworkConfiguration.addHiddenLayer(LayerType.CROSSCORRELATION, Initialization.UNIFORM_XAVIER_CONV, "filters = 1, filterSize = 3, isDepthSeparable = true");
//        neuralNetworkConfiguration.addHiddenLayer(LayerType.CROSSCORRELATION, Initialization.UNIFORM_XAVIER_CONV, "filters = 12, filterSize = 1, isDepthSeparable = false");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.BATCH_NORMALIZATION);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.RELU, "alpha = 0.01"));
        neuralNetworkConfiguration.addHiddenLayer(LayerType.AVERAGE_POOLING, "filterSize = 2, stride = 1");
//        neuralNetworkConfiguration.addHiddenLayer(LayerType.CROSSCORRELATION, Initialization.UNIFORM_XAVIER_CONV, "filters = 24, filterSize = 3, stride = 1");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.DSCROSSCORRELATION, Initialization.UNIFORM_XAVIER_CONV, "filters = 24, filterSize = 3, stride = 1");
//        neuralNetworkConfiguration.addHiddenLayer(LayerType.CROSSCORRELATION, Initialization.UNIFORM_XAVIER_CONV, "filters = 12, filterSize = 3, isDepthSeparable = true");
//        neuralNetworkConfiguration.addHiddenLayer(LayerType.CROSSCORRELATION, Initialization.UNIFORM_XAVIER_CONV, "filters = 24, filterSize = 1, isDepthSeparable = false");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.BATCH_NORMALIZATION);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.RELU, "alpha = 0.01"));
        neuralNetworkConfiguration.addHiddenLayer(LayerType.AVERAGE_POOLING, "filterSize = 2, stride = 1");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.FLATTEN);
//        neuralNetworkConfiguration.addHiddenLayer(LayerType.BATCH_NORMALIZATION);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.DENSE, "width = 100");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.RELU, "alpha = 0.01"));
        neuralNetworkConfiguration.addHiddenLayer(LayerType.DENSE, "width = " + outputSize);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.ACTIVATION, new ActivationFunction(UnaryFunctionType.SOFTMAX));
        neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
        neuralNetworkConfiguration.connectLayersSerially();

        NeuralNetwork neuralNetwork = new NeuralNetwork(neuralNetworkConfiguration);

        neuralNetwork.setOptimizer(OptimizationType.ADADELTA);
        return neuralNetwork; */

    }

    /**
     * Reads MNIST samples from CSV files.<br>
     * MNIST training set consist of 60000 samples and test set of 10000 samples.<br>
     * First column is assumed to be outputted digits (value 0 - 9).<br>
     * Next 784 (28x28) columns are assumed to be input digit (gray scale values 0 - 255).<br>
     *
     * @param trainSet if true training set file is read otherwise test set file is read.
     * @return encoded input and output pairs.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws FileNotFoundException throws exception if matrix cannot be read.
     */
    private static HashMap<Integer, HashMap<Integer, Matrix>> getMNISTData(boolean trainSet) throws MatrixException, FileNotFoundException {
        System.out.print("Loading " + (trainSet ? "training" : "test") + " data... ");
        HashSet<Integer> inputCols = new HashSet<>();
        HashSet<Integer> outputCols = new HashSet<>();
        for (int i = 1; i < 785; i++) inputCols.add(i);
        outputCols.add(0);
        String fileName = trainSet ? "<PATH>/mnist_train.csv" : "<PATH>/mnist_test.csv";
        HashMap<Integer, HashMap<Integer, Matrix>> data = ReadCSVFile.readFile(fileName, ",", inputCols, outputCols, 0, true, true, 28, 28, false, 0, 0);
        for (Matrix sample : data.get(0).values()) {
            sample.divideBy(255);
        }
        for (Integer sampleIndex : data.get(1).keySet()) {
            Matrix sample = data.get(1).get(sampleIndex);
            Matrix encodedSample = SMatrix.getOneHotVector(10, (int)sample.getValue(0,0, 0));
            data.get(1).put(sampleIndex, encodedSample);
        }

        System.out.println(" Done.");
        return data;
    }

}
