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
     * Default constructor for text sequence demo.
     *
     */
    public TextSeqDemo() {
    }

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
            HashMap<Integer, HashMap<Integer, Matrix>> data = getTextSeqData(numOfInputs, dictionaryIndexMapping);
            neuralNetwork = buildNeuralNetwork(data.get(0).get(0).getRows(), data.get(1).get(0).getRows());
//            neuralNetwork = Persistence.restoreNeuralNetwork(persistenceName);
            Persistence persistence = new Persistence(true, 100, neuralNetwork, persistenceName, true);
            neuralNetwork.setPersistence(persistence);
            neuralNetwork.verboseTraining(10);
            neuralNetwork.start();
            neuralNetwork.print();
            neuralNetwork.printExpressions();
            neuralNetwork.printGradients();
            neuralNetwork.resetDependencies(false);
            neuralNetwork.setTrainingData(new BasicSampler(new HashMap<>() {{ put(0, data.get(0)); }}, new HashMap<>() {{ put(0, data.get(1)); }},"randomOrder = false, randomStart = false, stepSize = 1, shuffleSamples = false, sampleSize = 100, numberOfIterations = 100"));
            while (neuralNetwork.getTotalIterations() < 100000) {
                neuralNetwork.train();
                System.out.println("Validating...");
                Matrix input = data.get(0).get(1);
                ArrayList<Matrix> encodedWords = input.getSubMatrices();
                int inputSize = encodedWords.get(0).size();
                for (int pos = 0; pos < 1000; pos++) {
                    Matrix finalInput = input;
                    Matrix nextEncodedWord = neuralNetwork.predictMatrix(new TreeMap<>() {{ put(0, finalInput);}}).get(0);
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
        NeuralNetworkConfiguration neuralNetworkConfiguration = new NeuralNetworkConfiguration();
        int inputLayerIndex = neuralNetworkConfiguration.addInputLayer("width = " + inputSize);
        int hiddenLayerIndex1 = neuralNetworkConfiguration.addHiddenLayer(LayerType.MINGRU, "width = 64");
        int hiddenLayerIndex2 = neuralNetworkConfiguration.addHiddenLayer(LayerType.MINGRU, "width = 64");
        int hiddenLayerIndex3 = neuralNetworkConfiguration.addHiddenLayer(LayerType.MINGRU, "width = 64, reversedInput = true");
        int hiddenLayerIndex4 = neuralNetworkConfiguration.addHiddenLayer(LayerType.MINGRU, "width = 64, reversedInput = true");
        int hiddenLayerIndex5 = neuralNetworkConfiguration.addHiddenLayer(LayerType.JOIN);
        int hiddenLayerIndex6 = neuralNetworkConfiguration.addHiddenLayer(LayerType.DOT_ATTENTION, "scaled = true");
        int hiddenLayerIndex7 = neuralNetworkConfiguration.addHiddenLayer(LayerType.LAYER_NORMALIZATION);
        int hiddenLayerIndex8 = neuralNetworkConfiguration.addHiddenLayer(LayerType.CONNECT);
        int hiddenLayerIndex9 = neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.GUMBEL_SOFTMAX), "width = " + outputSize);
        int outputLayerIndex = neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
        neuralNetworkConfiguration.connectLayers(inputLayerIndex, hiddenLayerIndex1);
        neuralNetworkConfiguration.connectLayers(hiddenLayerIndex1, hiddenLayerIndex2);
        neuralNetworkConfiguration.connectLayers(inputLayerIndex, hiddenLayerIndex3);
        neuralNetworkConfiguration.connectLayers(hiddenLayerIndex3, hiddenLayerIndex4);
        neuralNetworkConfiguration.connectLayers(hiddenLayerIndex2, hiddenLayerIndex5);
        neuralNetworkConfiguration.connectLayers(hiddenLayerIndex4, hiddenLayerIndex5);
        neuralNetworkConfiguration.connectLayers(hiddenLayerIndex5, hiddenLayerIndex6);
        neuralNetworkConfiguration.connectLayers(hiddenLayerIndex6, hiddenLayerIndex7);
        neuralNetworkConfiguration.connectLayers(inputLayerIndex, hiddenLayerIndex8);
        neuralNetworkConfiguration.connectLayers(hiddenLayerIndex7, hiddenLayerIndex8);
        neuralNetworkConfiguration.connectLayers(hiddenLayerIndex8, hiddenLayerIndex9);
        neuralNetworkConfiguration.connectLayers(hiddenLayerIndex9, outputLayerIndex);

        NeuralNetwork neuralNetwork = new NeuralNetwork(neuralNetworkConfiguration);

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
    private static HashMap<Integer, HashMap<Integer, Matrix>> getTextSeqData(int numOfInputs, HashMap<Integer, String> dictionaryIndexMapping) throws FileNotFoundException, MatrixException {
        return ReadTextFile.readFileAsBinaryEncoded("<PATH>/lorem_ipsum.txt", numOfInputs, 0, dictionaryIndexMapping);
    }

}
