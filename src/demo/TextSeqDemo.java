/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package demo;

import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.metrics.MetricsType;
import core.normalization.NormalizationType;
import core.optimization.*;
import core.preprocess.*;
import core.regularization.RegularizationType;
import utils.*;
import core.*;
import utils.matrix.*;
import utils.sampling.BasicSampler;

import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.LinkedHashMap;

/**
 * Creates recurrent neural network (RNN) that learns sequences of text.
 * RNN reads one or more characters as input and learns next character as output.
 * During validation phase RNN reproduces sequences it has learnt during training process.
 *
 */
public class TextSeqDemo {

    /**
     * Main function that reads data, executes learning process and replicates learnt text sequences.
     *
     * @param args input arguments (not used).
     */
    public static void main(String [] args) {

        NeuralNetwork neuralNetwork;
        try {
            String persistenceName = "<PATH>/TextSeqNN";
            int numOfInputs = 5;
            HashMap<Integer, LinkedHashMap<Integer, MMatrix>> data = getTextSeqData(numOfInputs);
            neuralNetwork = buildNeuralNetwork(data.get(0).get(0).get(0).getRows(), data.get(1).get(0).get(0).getRows());
//            neuralNetwork = Persistence.restoreNeuralNetwork(persistenceName);
            neuralNetwork.setTaskType(MetricsType.CLASSIFICATION);
            Persistence persistence = new Persistence(true, 100, neuralNetwork, persistenceName, true);
//            neuralNetwork.setPersistence(persistence);
            neuralNetwork.verboseTraining(10);
            neuralNetwork.start();
            neuralNetwork.print();
            neuralNetwork.printExpressions();
            neuralNetwork.printGradients();
            neuralNetwork.setTrainingData(new BasicSampler(data.get(0), data.get(1),"randomOrder = false, randomStart = false, stepSize = 100, shuffleSamples = false, sampleSize = 100, numberOfIterations = 100"));
            while (neuralNetwork.getTotalIterations() < 100000) {
                neuralNetwork.train();
                System.out.println("Validating...");
                int inputRows = data.get(0).get(0).get(0).getRows();
                int charSize = ReadTextFile.charSize();
                int numOfChars = inputRows / charSize;
                Matrix input = data.get(0).get(1).get(0);
                int[] letters = new int[numOfChars];
                int index = 0;
                for (int row = 0; row < inputRows; row++) {
                    if (input.getValue(row, 0) == 1) letters[index++] = row;
                }
                for (int pos = 0; pos < 1000; pos++) {
                    int nextLetter = neuralNetwork.predict(input).argmax()[0];
                    System.out.print((char)ReadTextFile.intToChar(nextLetter));
                    for (int charIndex = 0; charIndex < numOfChars - 1; charIndex++) {
                        letters[charIndex] = letters[charIndex + 1] - charSize;
                    }
                    letters[numOfChars - 1] = nextLetter + (numOfChars - 1) * charSize;
                    input = new DMatrix(inputRows, 1);
                    for (int charIndex = 0; charIndex < numOfChars; charIndex++) {
                        input.setValue(letters[charIndex], 0, 1);
                    }
                }
                System.out.println();
            }
            neuralNetwork.stop();
        }
        catch (Exception exception) {
            exception.printStackTrace();
            System.exit(-1);
        }
    }

    /**
     * Builds recurrent neural network (GRU) instance.
     *
     * @param inputSize input size of neural network (digits as one hot encoded in sequence).
     * @param outputSize output size of neural network (digits as one hot encoded in sequence).
     * @return neural network instance.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = " + inputSize);
        neuralNetwork.addHiddenLayer(LayerType.GRAVESLSTM, "width = 100");
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX), "width = " + outputSize);
        neuralNetwork.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.ADAM);
        neuralNetwork.addNormalizer(2, NormalizationType.LAYER_NORMALIZATION);
//        neuralNetwork.addRegularizer(1, RegularizationType.DROPOUT, "probability = 0.1");
        neuralNetwork.addRegularizer(RegularizationType.WEIGHT_NOISING);
        return neuralNetwork;
    }

    /**
     * Function that reads text file and one hot encodes it for inputs and outputs.
     * Output is usually next character in sequence following input characters.
     *
     * @return encoded inputs and outputs.
     * @throws FileNotFoundException throws exception if file is not found.
     */
    private static HashMap<Integer, LinkedHashMap<Integer, MMatrix>> getTextSeqData(int numOfInputs) throws FileNotFoundException {
        return ReadTextFile.readFile("<PATH>/lorem_ipsum.txt", numOfInputs, 1, numOfInputs, 0);
    }

}
