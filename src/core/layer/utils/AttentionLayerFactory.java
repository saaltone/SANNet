/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.utils;

import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.network.NeuralNetworkConfiguration;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunctionType;

/**
 * Defines factory class to build attention layer components including transformer.
 *
 */
public class AttentionLayerFactory {

    /**
     * Default constructor for attention layer factory.
     *
     */
    public AttentionLayerFactory() {
    }

    /**
     * Builds transformer.
     *
     * @param neuralNetworkConfiguration neural network configuration.
     * @param numberOfInputs             number of inputs.
     * @param inputWidth                 input width.
     * @param inputHeight                input height.
     * @param inputDepth                 input depth.
     * @param numberOfAttentionBlocks    number of attention blocks.
     * @param flattenOutput              if true output is flattened after last attention block.
     * @return output layer index.
     * @throws DynamicParamException  throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException        throws exception if custom function is attempted to be created with this constructor.
     */
    public static int buildTransformer(NeuralNetworkConfiguration neuralNetworkConfiguration, int numberOfInputs, int inputWidth, int inputHeight, int inputDepth, int numberOfAttentionBlocks, boolean flattenOutput) throws NeuralNetworkException, MatrixException, DynamicParamException {
        return buildTransformer(neuralNetworkConfiguration, numberOfInputs, inputWidth, inputHeight, inputDepth, numberOfAttentionBlocks, 0.1, true, flattenOutput);
    }

    /**
     * Builds transformer.
     *
     * @param neuralNetworkConfiguration neural network configuration.
     * @param numberOfInputs             number of inputs.
     * @param inputWidth                 input width.
     * @param inputHeight                input height.
     * @param inputDepth                 input depth.
     * @param numberOfAttentionBlocks    number of attention blocks.
     * @param dropoutProbability         dropout probability.
     * @param normalize                  if true normalized layer output otherwise not.
     * @param flattenOutput              if true output is flattened after last attention block.
     * @return output layer index.
     * @throws DynamicParamException  throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException        throws exception if custom function is attempted to be created with this constructor.
     */
    public static int buildTransformer(NeuralNetworkConfiguration neuralNetworkConfiguration, int numberOfInputs, int inputWidth, int inputHeight, int inputDepth, int numberOfAttentionBlocks, double dropoutProbability, boolean normalize, boolean flattenOutput) throws NeuralNetworkException, MatrixException, DynamicParamException {
        int attentionBlockIndex =  buildInputAttentionBlock(neuralNetworkConfiguration, numberOfInputs, inputWidth, inputHeight, inputDepth, dropoutProbability, normalize, numberOfAttentionBlocks == 1 && flattenOutput);
        for (int blockIndex = 0; blockIndex < numberOfAttentionBlocks - 1; blockIndex++) {
            attentionBlockIndex = buildAttentionBlock(neuralNetworkConfiguration, attentionBlockIndex, dropoutProbability, normalize, blockIndex == numberOfAttentionBlocks - 2 && flattenOutput);
        }
        return attentionBlockIndex;
    }

    /**
     * Builds input attention block.
     *
     * @param neuralNetworkConfiguration neural network configuration.
     * @param numberOfInputs             number of inputs.
     * @param inputWidth                 input width.
     * @param inputHeight                input height.
     * @param inputDepth                 input depth.
     * @param flattenOutput              if true output of attention block is flattened.
     * @return output layer index.
     * @throws DynamicParamException  throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException        throws exception if custom function is attempted to be created with this constructor.
     */
    public static int buildInputAttentionBlock(NeuralNetworkConfiguration neuralNetworkConfiguration, int numberOfInputs, int inputWidth, int inputHeight, int inputDepth, boolean flattenOutput) throws NeuralNetworkException, MatrixException, DynamicParamException {
        return buildInputAttentionBlock(neuralNetworkConfiguration, numberOfInputs, inputWidth, inputHeight, inputDepth, 0.1, true, flattenOutput);
    }

    /**
     * Builds input attention block.
     *
     * @param neuralNetworkConfiguration neural network configuration.
     * @param numberOfInputs             number of inputs.
     * @param inputWidth                 input width.
     * @param inputHeight                input height.
     * @param inputDepth                 input depth.
     * @param dropoutProbability         dropout probability.
     * @param normalize                  if true normalized layer output otherwise not.
     * @param flattenOutput              if true output of attention block is flattened.
     * @return output layer index.
     * @throws DynamicParamException  throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException        throws exception if custom function is attempted to be created with this constructor.
     */
    public static int buildInputAttentionBlock(NeuralNetworkConfiguration neuralNetworkConfiguration, int numberOfInputs, int inputWidth, int inputHeight, int inputDepth, double dropoutProbability, boolean normalize, boolean flattenOutput) throws NeuralNetworkException, MatrixException, DynamicParamException {
        // Input layers with positional encoding.
        if (numberOfInputs > 1) {
            int[] inputLayerIndices = new int[numberOfInputs];
            for (int inputIndex = 0; inputIndex < numberOfInputs; inputIndex++) {
                int inputLayerIndex = neuralNetworkConfiguration.addInputLayer("width = " + inputWidth + ", height = " + inputHeight + ", depth = " + inputDepth);
                int positionalEmbeddingLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.POSITIONAL_ENCODING, "heightWisePositionalEncoding = false, positionIndex = " + inputIndex);
                neuralNetworkConfiguration.connectLayers(inputLayerIndex, positionalEmbeddingLayerIndex);
                inputLayerIndices[inputIndex] = positionalEmbeddingLayerIndex;
            }

            int attentionLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.DOT_ATTENTION);
            for (int inputLayerIndex : inputLayerIndices) neuralNetworkConfiguration.connectLayers(inputLayerIndex, attentionLayerIndex);

            return buildAttentionBlock(neuralNetworkConfiguration, true, attentionLayerIndex, dropoutProbability, normalize, flattenOutput);
        }
        else {
            int inputLayerIndex = neuralNetworkConfiguration.addInputLayer("width = " + inputWidth + ", height = " + inputHeight + ", depth = " + inputDepth);
            int positionalEncodingIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.POSITIONAL_ENCODING, "heightWisePositionalEncoding = true, embeddingSize = " + inputWidth);
            neuralNetworkConfiguration.connectLayers(inputLayerIndex, positionalEncodingIndex);
            return buildAttentionBlock(neuralNetworkConfiguration, false, positionalEncodingIndex, dropoutProbability, normalize, flattenOutput);
        }
    }

    /**
     * Builds attention block
     *
     * @param neuralNetworkConfiguration neural network configuration.
     * @param inputLayerIndex            input layer index.
     * @param dropoutProbability         dropout probability.
     * @param normalize                  if true normalized layer output otherwise not.
     * @param flattenOutput              if true output of attention block is flattened.
     * @return output layer index.
     * @throws DynamicParamException  throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException        throws exception if custom function is attempted to be created with this constructor.
     */
    public static int buildAttentionBlock(NeuralNetworkConfiguration neuralNetworkConfiguration, int inputLayerIndex, double dropoutProbability, boolean normalize, boolean flattenOutput) throws MatrixException, NeuralNetworkException, DynamicParamException {
        return buildAttentionBlock(neuralNetworkConfiguration, false, inputLayerIndex, dropoutProbability, normalize, flattenOutput);
    }

    /**
     * Builds attention block
     *
     * @param neuralNetworkConfiguration neural network configuration.
     * @param asInputAttention           if true input attention block otherwise normal attention block.
     * @param inputLayerIndex            input layer index.
     * @param dropoutProbability         dropout probability.
     * @param normalize                  if true normalized layer output otherwise not.
     * @param flattenOutput              if true output of attention block is flattened.
     * @return output layer index.
     * @throws DynamicParamException  throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException        throws exception if custom function is attempted to be created with this constructor.
     */
    private static int buildAttentionBlock(NeuralNetworkConfiguration neuralNetworkConfiguration, boolean asInputAttention, int inputLayerIndex, double dropoutProbability, boolean normalize, boolean flattenOutput) throws MatrixException, NeuralNetworkException, DynamicParamException {
        int inputLayerIndex0;
        if (!asInputAttention) {
            int attentionLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.DOT_ATTENTION);
            neuralNetworkConfiguration.connectLayers(inputLayerIndex, attentionLayerIndex);
            inputLayerIndex0 = attentionLayerIndex;
        } else inputLayerIndex0 = inputLayerIndex;

        int nextLayerIndex0;
        if (dropoutProbability > 0) {
            int dropoutLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.DROPOUT, "probability = " + dropoutProbability);
            neuralNetworkConfiguration.connectLayers(inputLayerIndex0, dropoutLayerIndex);
            nextLayerIndex0 = dropoutLayerIndex;
        }
        else nextLayerIndex0 = inputLayerIndex0;

        int addLayerIndex1 = neuralNetworkConfiguration.addHiddenLayer(LayerType.ADD);
        neuralNetworkConfiguration.connectLayers(nextLayerIndex0, addLayerIndex1);
        neuralNetworkConfiguration.connectLayers(inputLayerIndex, addLayerIndex1);

        int nextLayerIndex1;
        if (normalize) {
            int normalizationLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.LAYER_NORMALIZATION);
            neuralNetworkConfiguration.connectLayers(addLayerIndex1, normalizationLayerIndex);
            nextLayerIndex1 = normalizationLayerIndex;
        }
        else nextLayerIndex1 = addLayerIndex1;

        int feedforwardLayerIndex1 = neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU));
        neuralNetworkConfiguration.connectLayers(nextLayerIndex1, feedforwardLayerIndex1);

        int nextLayerIndex2;
        if (dropoutProbability > 0) {
            int dropoutLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.DROPOUT, "probability = " + dropoutProbability);
            neuralNetworkConfiguration.connectLayers(feedforwardLayerIndex1, dropoutLayerIndex);
            nextLayerIndex2 = dropoutLayerIndex;
        }
        else nextLayerIndex2 = feedforwardLayerIndex1;

        int feedforwardLayerIndex2 = neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU));
        neuralNetworkConfiguration.connectLayers(nextLayerIndex2, feedforwardLayerIndex2);

        int addLayerIndex2 = neuralNetworkConfiguration.addHiddenLayer(LayerType.ADD);
        neuralNetworkConfiguration.connectLayers(nextLayerIndex1, addLayerIndex2);
        neuralNetworkConfiguration.connectLayers(feedforwardLayerIndex2, addLayerIndex2);

        int nextLayerIndex3;
        if (normalize) {
            int normalizationLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.LAYER_NORMALIZATION);
            neuralNetworkConfiguration.connectLayers(addLayerIndex2, normalizationLayerIndex);
            nextLayerIndex3 = normalizationLayerIndex;
        }
        else nextLayerIndex3 = addLayerIndex2;

        if (flattenOutput) {
            int flattenLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.FLATTEN);
            neuralNetworkConfiguration.connectLayers(nextLayerIndex3, flattenLayerIndex);
            return flattenLayerIndex;
        }
        else return nextLayerIndex3;

    }

}
