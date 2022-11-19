/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package demo;

import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.network.NeuralNetwork;
import core.network.NeuralNetworkException;
import core.network.Persistence;
import core.optimization.*;
import core.preprocess.*;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.sampling.BasicSampler;

import java.io.FileNotFoundException;
import java.util.*;

/**
 * Creates recurrent neural network (RNN) that learns text sequences.<br>
 * RNN reads one or more characters as input and learns next character as output.<br>
 * During validation phase RNN reproduces sequences it has learnt during training process.<br>
 *
 */
public class TextSeqDemo {

    /**
     * Main function that reads data, executes learning process and replicates learnt text sequences.
     *
     * @param args input arguments (not used).
     */
    public static void main(String [] args) {

        int numOfInputs = 5;

        NeuralNetwork neuralNetwork;
        try {
            String persistenceName = "<PATH>/TextSeqNN";
            HashMap<Integer, String> dictionaryIndexMapping = new HashMap<>();
            HashMap<Integer, HashMap<Integer, MMatrix>> data = getTextSeqData(numOfInputs, dictionaryIndexMapping);
            neuralNetwork = buildNeuralNetwork(data.get(0).get(0).get(0).getRows(), data.get(1).get(0).get(0).getRows());
//            neuralNetwork = Persistence.restoreNeuralNetwork(persistenceName);
            Persistence persistence = new Persistence(true, 100, neuralNetwork, persistenceName, true);
            neuralNetwork.setPersistence(persistence);
            neuralNetwork.verboseTraining(10);
            neuralNetwork.start();
            neuralNetwork.print();
            neuralNetwork.printExpressions();
            neuralNetwork.printGradients();
            neuralNetwork.setTrainingData(new BasicSampler(data.get(0), data.get(1),"randomOrder = false, randomStart = false, stepSize = 1, shuffleSamples = false, sampleSize = 100, numberOfIterations = 100"));
            while (neuralNetwork.getTotalIterations() < 100000) {
                neuralNetwork.train();
                System.out.println("Validating...");
                Matrix input = data.get(0).get(1).get(0);
                ArrayList<Matrix> encodedWords = input.getSubMatrices();
                int inputSize = encodedWords.get(0).size();
                for (int pos = 0; pos < 1000; pos++) {
                    Matrix nextEncodedWord = neuralNetwork.predict(input);
                    int wordIndex = nextEncodedWord.argmax()[0];
                    String currentWord = dictionaryIndexMapping.getOrDefault(wordIndex, "???");
                    System.out.print(currentWord + " ");
                    nextEncodedWord = ComputableMatrix.encodeToBitColumnVector(wordIndex, inputSize);
                    for (int index = 0; index < encodedWords.size() - 1; index++) {
                        encodedWords.set(index, encodedWords.get(index + 1));
                    }
                    encodedWords.set(encodedWords.size() - 1, nextEncodedWord);
                    input = new JMatrix(encodedWords, true);
                    encodedWords = input.getSubMatrices();
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
        neuralNetwork.addHiddenLayer(LayerType.BIMINGRU, "width = 64");
        neuralNetwork.addHiddenLayer(LayerType.BIMINGRU, "width = 64");
        neuralNetwork.addHiddenLayer(LayerType.LAYER_NORMALIZATION);
        neuralNetwork.addHiddenLayer(LayerType.CONNECTOR, "inputLayers = [0]");
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.GUMBEL_SOFTMAX), "width = " + outputSize);
        neuralNetwork.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.ADAM);
        return neuralNetwork;
    }

    /**
     * Function that reads text file and one hot encodes it for inputs and outputs.
     * Output is usually next character in sequence following input characters.
     *
     * @return encoded inputs and outputs.
     * @throws FileNotFoundException throws exception if file is not found.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private static HashMap<Integer, HashMap<Integer, MMatrix>> getTextSeqData(int numOfInputs, HashMap<Integer, String> dictionaryIndexMapping) throws FileNotFoundException, MatrixException {
        return ReadTextFile.readFileAsBinaryEncoded("<PATH>/lorem_ipsum.txt", numOfInputs, 0, dictionaryIndexMapping);
    }

}
