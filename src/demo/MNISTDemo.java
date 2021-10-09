/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package demo;

import core.network.NeuralNetwork;
import core.network.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.metrics.Metrics;
import core.metrics.MetricsType;
import core.normalization.NormalizationType;
import core.optimization.OptimizationType;
import core.preprocess.ReadCSVFile;
import core.regularization.EarlyStopping;
import utils.DynamicParamException;
import utils.Persistence;
import utils.Sequence;
import utils.matrix.*;
import utils.sampling.BasicSampler;

import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;

/**
 * Class that implements learning and prediction of MNIST digits by using Convolutional Neural Network (CNN).<br>
 * Implementation reads training and test samples from CSV file and then executes learning process.<br>
 *
 */
public class MNISTDemo {

    /**
     * Main function for demo.<br>
     * Function reads samples from files, build CNN and executed training, validation and testing.<br>
     *
     * @param args input arguments (not used).
     */
    public static void main(String [] args) {

        NeuralNetwork neuralNetwork;
        try {
            HashMap<Integer, LinkedHashMap<Integer, MMatrix>> trainMNIST = getMNISTData(true);
            HashMap<Integer, LinkedHashMap<Integer, MMatrix>> testMNIST = getMNISTData(false);

            neuralNetwork = buildNeuralNetwork(trainMNIST.get(0).get(0).get(0).getRows(), trainMNIST.get(1).get(0).get(0).getRows());

            String persistenceName = "<PATH>/MNIST_NN";
//            neuralNetwork = Persistence.restoreNeuralNetwork(persistenceName);

            Persistence persistence = new Persistence(true, 100, neuralNetwork, persistenceName, true);
            neuralNetwork.setPersistence(persistence);

            neuralNetwork.setTaskType(MetricsType.CLASSIFICATION, false);
            neuralNetwork.verboseTraining(10);
            neuralNetwork.setAutoValidate(100);
            neuralNetwork.verboseValidation();
            neuralNetwork.setTrainingEarlyStopping(new EarlyStopping());

            neuralNetwork.start();

            neuralNetwork.setTrainingData(new BasicSampler(trainMNIST.get(0), trainMNIST.get(1), "randomOrder = true, shuffleSamples = true, sampleSize = 16, numberOfIterations = 5625"));
            neuralNetwork.setValidationData(new BasicSampler(testMNIST.get(0), testMNIST.get(1), "randomOrder = true, shuffleSamples = true, sampleSize = 10"));

            neuralNetwork.print();
            neuralNetwork.printExpressions();
            neuralNetwork.printGradients();

            System.out.println("Training...");
            neuralNetwork.train();

            System.out.println("Predicting...");

            Metrics predictionMetrics = new Metrics(MetricsType.CLASSIFICATION);
            for (int index = 0; index < 100; index++) {
                Sequence input = new Sequence(1);
                Sequence output = new Sequence(1);
                for (int index1 = 0; index1 < 100; index1++) {
                    input.put(index1, testMNIST.get(0).get(index * 100 + index1));
                    output.put(index1, testMNIST.get(1).get(index * 100 + index1));
                }
                Sequence predict = neuralNetwork.predict(input);
                if (index == 0) {
                    System.out.println("Printing out first 100 predictions...");
                    for (int index1 = 0; index1 < 100; index1++) {
                        int[] trueIndex = output.get(index1).get(0).argmax();
                        int[] predictIndex = predict.get(index1).get(0).argmax();
                        System.out.println("True label: " + trueIndex[0] + ", Predicted label: " + predictIndex[0]);
                    }
                }
                predictionMetrics.report(predict, output);
                predictionMetrics.store(index, false);
            }
            predictionMetrics.printReport();

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
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = 28, height = 28");
        neuralNetwork.addHiddenLayer(LayerType.CONVOLUTIONAL, new ActivationFunction(UnaryFunctionType.RELU), Initialization.UNIFORM_XAVIER_CONV, "filters = 12, filterSize = 3, stride = 1, asConvolution = false, asWinogradConvolution = false");
        neuralNetwork.addHiddenLayer(LayerType.CONVOLUTIONAL, new ActivationFunction(UnaryFunctionType.RELU), Initialization.UNIFORM_XAVIER_CONV, "filters = 24, filterSize = 3, stride = 1, asConvolution = false, asWinogradConvolution = false");
        neuralNetwork.addHiddenLayer(LayerType.POOLING, "filterSize = 2, stride = 1, avgPool = false");
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = 100");
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX), "width = " + outputSize);
        neuralNetwork.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.ADADELTA);
        neuralNetwork.addNormalizer(4, NormalizationType.BATCH_NORMALIZATION);
        return neuralNetwork;
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
    private static HashMap<Integer, LinkedHashMap<Integer, MMatrix>> getMNISTData(boolean trainSet) throws MatrixException, FileNotFoundException {
        System.out.print("Loading " + (trainSet ? "training" : "test") + " data... ");
        HashSet<Integer> inputCols = new HashSet<>();
        HashSet<Integer> outputCols = new HashSet<>();
        for (int i = 1; i < 785; i++) inputCols.add(i);
        outputCols.add(0);
        String fileName = trainSet ? "<PATH>/mnist_train.csv" : "<PATH>/mnist_test.csv";
        HashMap<Integer, LinkedHashMap<Integer, MMatrix>> data = ReadCSVFile.readFile(fileName, ",", inputCols, outputCols, 0, true, true, 28, 28, false, 0, 0);
        for (MMatrix sample : data.get(0).values()) {
            for (Matrix entry : sample.values()) {
                entry.divide(255, entry);
            }
        }
        for (MMatrix sample : data.get(1).values()) {
            for (Integer entryIndex : sample.keySet()) {
                int value = (int)sample.get(entryIndex).getValue(0,0);
                Matrix output = new SMatrix(10, 1);
                output.setValue(value, 0, 1);
                sample.put(entryIndex, output);
            }
        }
        System.out.println(" Done.");
        return data;
    }

}
