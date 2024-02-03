/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package demo;

import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.layer.utils.AttentionLayerFactory;
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
        NeuralNetwork neuralNetwork;
        try {
            String persistenceName = "<PATH>/TextSeqNN";
            int numOfInputs = 50;
            boolean joinInputsVertically = false;
            HashMap<Integer, String> dictionaryIndexMapping = new HashMap<>();
            HashMap<Integer, HashMap<Integer, Matrix>> data = getTextSeqData(numOfInputs, dictionaryIndexMapping, joinInputsVertically);
            neuralNetwork = buildNeuralNetwork(data.get(0).get(0).getRows(), data.get(1).get(0).getRows(), joinInputsVertically ? 1 : numOfInputs);
//            neuralNetwork = Persistence.restoreNeuralNetwork(persistenceName);
            Persistence persistence = new Persistence(true, 100, neuralNetwork, persistenceName, true);
            neuralNetwork.setPersistence(persistence);
            neuralNetwork.verboseTraining(10);
            neuralNetwork.setShowTrainingMetrics(true);
            neuralNetwork.start();
            neuralNetwork.print();
            neuralNetwork.printExpressions();
            neuralNetwork.printGradients();
            neuralNetwork.resetDependencies(false);
            neuralNetwork.setTrainingData(new BasicSampler(new HashMap<>() {{ put(0, data.get(0)); }}, new HashMap<>() {{ put(0, data.get(1)); }},"randomOrder = false, randomStart = false, stepSize = 1, shuffleSamples = false, sampleSize = 100, numberOfIterations = 100"));
            while (neuralNetwork.getTotalTrainingIterations() < 100000) {
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
                    nextEncodedWord = AbstractMatrix.encodeValueToBitColumnVector(wordIndex, inputSize);
                    for (int index = 0; index < encodedWords.size() - 1; index++) {
                        encodedWords.set(index, encodedWords.get(index + 1));
                    }
                    encodedWords.set(encodedWords.size() - 1, nextEncodedWord);
                    input = new JMatrix(encodedWords, joinInputsVertically);
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
     * @param inputSize   input size of neural network (digits as one hot encoded in sequence).
     * @param outputSize  output size of neural network (digits as one hot encoded in sequence).
     * @param layerHeight layer height.
     * @return neural network instance.
     * @throws DynamicParamException  throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException        throws exception if custom function is attempted to be created with this constructor.
     */
    private static NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize, int layerHeight) throws DynamicParamException, NeuralNetworkException, MatrixException {
        double tau = 0.8;
        int numberOfAttentionBlocks = 2;
        double dropoutProbability = 0.1;
        boolean normalize = true;

        NeuralNetworkConfiguration neuralNetworkConfiguration = new NeuralNetworkConfiguration();
        int attentionLayer = AttentionLayerFactory.buildTransformer(neuralNetworkConfiguration, 1, inputSize, layerHeight, 1, numberOfAttentionBlocks, dropoutProbability, normalize, true);
        int feedforwardLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX, "tau = " + tau), "width = " + outputSize);
        neuralNetworkConfiguration.connectLayers(attentionLayer, feedforwardLayerIndex);
        int outputLayerIndex = neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.CROSS_ENTROPY);
        neuralNetworkConfiguration.connectLayers(feedforwardLayerIndex, outputLayerIndex);

        NeuralNetwork neuralNetwork = new NeuralNetwork(neuralNetworkConfiguration);

        neuralNetwork.setOptimizer(OptimizationType.ADAM);
        return neuralNetwork;
    }

    /**
     * Function that reads text file and one hot encodes it for inputs and outputs.
     * Output is usually next character in sequence following input characters.
     *
     * @param numOfInputs number of inputs.
     * @param dictionaryIndexMapping dictionary index mapping
     * @param joinInputsVertically if true joins input vertically otherwise horizontally.
     * @return encoded inputs and outputs.
     * @throws FileNotFoundException throws exception if file is not found.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private static HashMap<Integer, HashMap<Integer, Matrix>> getTextSeqData(int numOfInputs, HashMap<Integer, String> dictionaryIndexMapping, boolean joinInputsVertically) throws FileNotFoundException, MatrixException {
        return ReadTextFile.readFileAsBinaryEncoded("<PATH>/lorem_ipsum.txt", numOfInputs, 0, dictionaryIndexMapping, joinInputsVertically);
    }

}
