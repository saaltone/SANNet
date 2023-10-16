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

import java.util.Stack;

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
     * Builds input attention module
     *
     * @param neuralNetworkConfiguration neural network configuration
     * @param inputSize input size
     * @param numberOfInputs number of inputs
     * @param isEncoder is true attention module is for encoder otherwise for decoder
     * @return attention layer index
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public static int buildInputAttentionModule(NeuralNetworkConfiguration neuralNetworkConfiguration, int inputSize, int numberOfInputs, boolean isEncoder) throws MatrixException, NeuralNetworkException, DynamicParamException {
        // Input layers with positional encoding.
        int[] inputLayerIndices = new int[numberOfInputs];
        for (int inputIndex = 0; inputIndex < numberOfInputs; inputIndex++) {
            int inputLayerIndex = neuralNetworkConfiguration.addInputLayer(isEncoder ? 1 : 0, "width = " + inputSize);
            int positionalEmbeddingLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.POSITIONAL_ENCODING, "positionIndex = " + inputIndex);
            neuralNetworkConfiguration.connectLayers(inputLayerIndex, positionalEmbeddingLayerIndex);
            inputLayerIndices[inputIndex] = positionalEmbeddingLayerIndex;
        }

        // Attention layer for input information.
        int attentionLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.DOT_ATTENTION, "scaled = true");
        int addLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.ADD);
        for (int inputIndex = 0; inputIndex < numberOfInputs; inputIndex++) {
            neuralNetworkConfiguration.connectLayers(inputLayerIndices[inputIndex], attentionLayerIndex);
            neuralNetworkConfiguration.connectLayers(inputLayerIndices[inputIndex], addLayerIndex);
        }
        neuralNetworkConfiguration.connectLayers(attentionLayerIndex, addLayerIndex);

        // Normalization layer
        int normLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.LAYER_NORMALIZATION);
        neuralNetworkConfiguration.connectLayers(addLayerIndex, normLayerIndex);

        return normLayerIndex;
    }

    /**
     * Builds input attention block
     *
     * @param neuralNetworkConfiguration neural network configuration
     * @param inputSize input size
     * @param numberOfInputs number of inputs
     * @param isEncoder is true attention module is for encoder otherwise for decoder
     * @param decoderInputSize decoder input size
     * @return attention layer index
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public static int buildInputAttentionBlock(NeuralNetworkConfiguration neuralNetworkConfiguration, int inputSize, int numberOfInputs, boolean isEncoder, int decoderInputSize) throws MatrixException, NeuralNetworkException, DynamicParamException {
        return buildFeedforwardModule(neuralNetworkConfiguration, buildInputAttentionModule(neuralNetworkConfiguration, inputSize, numberOfInputs, isEncoder), -1, isEncoder ? decoderInputSize : -1);
    }

    /**
     * Builds decoder input attention block
     *
     * @param neuralNetworkConfiguration neural network configuration
     * @param inputSize input size
     * @param numberOfInputs number of inputs
     * @return attention layer index
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public static int buildDecoderInputAttentionBlock(NeuralNetworkConfiguration neuralNetworkConfiguration, int inputSize, int numberOfInputs) throws MatrixException, NeuralNetworkException, DynamicParamException {
        return buildInputAttentionBlock(neuralNetworkConfiguration, inputSize, numberOfInputs, false, -1);
    }

    /**
     * Builds encoder input attention block
     *
     * @param neuralNetworkConfiguration neural network configuration
     * @param inputSize input size
     * @param numberOfInputs number of inputs
     * @param decoderInputSize decoder input size
     * @return attention layer index
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public static int buildEncoderInputAttentionBlock(NeuralNetworkConfiguration neuralNetworkConfiguration, int inputSize, int numberOfInputs, int decoderInputSize) throws MatrixException, NeuralNetworkException, DynamicParamException {
        return buildInputAttentionBlock(neuralNetworkConfiguration, inputSize, numberOfInputs, true, decoderInputSize);
    }

    /**
     * Builds attention block
     *
     * @param neuralNetworkConfiguration neural network configuration
     * @param inputLayerIndex            input layer index
     * @param feedforwardLayerWidth      feedforward layer width
     * @return attention module index.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public static int buildAttentionBlock(NeuralNetworkConfiguration neuralNetworkConfiguration, int inputLayerIndex, int feedforwardLayerWidth) throws NeuralNetworkException, MatrixException, DynamicParamException {
        return buildFeedforwardModule(neuralNetworkConfiguration, buildAttentionModule(neuralNetworkConfiguration, inputLayerIndex, -1), feedforwardLayerWidth, -1);
    }

    /**
     * Builds attention module
     *
     * @param neuralNetworkConfiguration      neural network configuration
     * @param inputLayerIndex                 input layer index
     * @param encoderInputLayerIndex          (optional) encoder input index
     * @return attention module index.
     * @throws DynamicParamException  throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException        throws exception if custom function is attempted to be created with this constructor.
     */
    public static int buildAttentionModule(NeuralNetworkConfiguration neuralNetworkConfiguration, int inputLayerIndex, int encoderInputLayerIndex) throws NeuralNetworkException, MatrixException, DynamicParamException {
        // Attention layer
        int attentionLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.DOT_ATTENTION, "scaled = true");
        neuralNetworkConfiguration.connectLayers(inputLayerIndex, attentionLayerIndex);
        if (encoderInputLayerIndex != -1) neuralNetworkConfiguration.connectLayers(encoderInputLayerIndex, attentionLayerIndex);

        // Additive layer
        int addLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.ADD);
        neuralNetworkConfiguration.connectLayers(inputLayerIndex, addLayerIndex);
        neuralNetworkConfiguration.connectLayers(attentionLayerIndex, addLayerIndex);

        // Normalization layer
        int normLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.LAYER_NORMALIZATION);
        neuralNetworkConfiguration.connectLayers(addLayerIndex, normLayerIndex);

        return normLayerIndex;
    }

    /**
     * Builds attention block
     *
     * @param neuralNetworkConfiguration neural network configuration
     * @param inputIndex                 input index
     * @param encoderInputIndex          (optional) encoder input index
     * @param feedforwardLayerWidth      feedforward layer width
     * @return attention module index.
     * @throws DynamicParamException  throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException        throws exception if custom function is attempted to be created with this constructor.
     */
    public static int buildAttentionBlock(NeuralNetworkConfiguration neuralNetworkConfiguration, int inputIndex, int encoderInputIndex, int feedforwardLayerWidth) throws NeuralNetworkException, MatrixException, DynamicParamException {
        return buildFeedforwardModule(neuralNetworkConfiguration, buildAttentionModule(neuralNetworkConfiguration, inputIndex, encoderInputIndex), feedforwardLayerWidth, -1);
    }

    /**
     * Builds feedforward module
     *
     * @param neuralNetworkConfiguration neural network configuration
     * @param inputIndex input index
     * @param layerWidth layer width
     * @param layerOutputWidth layer output width
     * @return feedforward module layer index.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public static int buildFeedforwardModule(NeuralNetworkConfiguration neuralNetworkConfiguration, int inputIndex, int layerWidth, int layerOutputWidth) throws MatrixException, NeuralNetworkException, DynamicParamException {
        // Feedforward layer
        int feedforwardLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), layerWidth > 0 ? "width = " + layerWidth : null);
        neuralNetworkConfiguration.connectLayers(inputIndex, feedforwardLayerIndex);

        // Connect layer
        int addLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.CONNECT, layerOutputWidth > 0 ? "width = " + layerOutputWidth : null);
        neuralNetworkConfiguration.connectLayers(inputIndex, addLayerIndex);
        neuralNetworkConfiguration.connectLayers(feedforwardLayerIndex, addLayerIndex);

        // Normalization layer
        int normLayerIndex = neuralNetworkConfiguration.addHiddenLayer(LayerType.LAYER_NORMALIZATION);
        neuralNetworkConfiguration.connectLayers(addLayerIndex, normLayerIndex);

        return normLayerIndex;
    }

    /**
     * Build decoder only transformer.
     *
     * @param neuralNetworkConfiguration neural network configuration.
     * @param decoderInputSize decoder input size.
     * @param numberOfDecoderInputs number of decoder outputs.
     * @param feedforwardLayerWidth feedforward layer width.
     * @param numberOfAttentionBlocks number of attention blocks.
     * @return transformer layer id
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public static int buildTransformer(NeuralNetworkConfiguration  neuralNetworkConfiguration, int decoderInputSize, int numberOfDecoderInputs, int feedforwardLayerWidth, int numberOfAttentionBlocks) throws MatrixException, NeuralNetworkException, DynamicParamException {
        return buildTransformer(neuralNetworkConfiguration, decoderInputSize, numberOfDecoderInputs, -1, -1, feedforwardLayerWidth, numberOfAttentionBlocks);
    }

    /**
     * Builds transformer with option to have encoder decoder or decoder only.
     *
     * @param neuralNetworkConfiguration neural network configuration.
     * @param encoderInputSize encoder input size. If set to -1 results into decoder only transformer.
     * @param numberOfEncoderInputs number of encoder inputs. If set to -1 results into decoder only transformer.
     * @param decoderInputSize decoder output size.
     * @param numberOfDecoderInputs number of decoder inputs.
     * @param feedforwardWidth feedforward layer width.
     * @param numberOfAttentionBlocks number of attention blocks.
     * @return transformer layer id.
     * @throws DynamicParamException throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public static int buildTransformer(NeuralNetworkConfiguration  neuralNetworkConfiguration, int decoderInputSize, int numberOfDecoderInputs, int encoderInputSize, int numberOfEncoderInputs, int feedforwardWidth, int numberOfAttentionBlocks) throws MatrixException, NeuralNetworkException, DynamicParamException {
        boolean decoderOnly = (encoderInputSize == -1 && numberOfEncoderInputs == -1);

        // Encoders and decoders
        int decoderLayerIndex = buildDecoderInputAttentionBlock(neuralNetworkConfiguration, decoderInputSize, numberOfDecoderInputs);
        int encoderLayerIndex = decoderOnly ? -1 : buildEncoderInputAttentionBlock(neuralNetworkConfiguration, encoderInputSize, numberOfEncoderInputs, decoderInputSize);

        // (Optional) attention block for connecting encoder and decoder.
        int combinedAttentionIndex = decoderOnly ? decoderLayerIndex : buildAttentionBlock(neuralNetworkConfiguration, decoderLayerIndex, encoderLayerIndex, feedforwardWidth);

        // Attention blocks
        for (int index = 0; index < numberOfAttentionBlocks; index++) {
            combinedAttentionIndex = buildAttentionBlock(neuralNetworkConfiguration, combinedAttentionIndex, feedforwardWidth);
        }

        return combinedAttentionIndex;
    }

    /**
     * Builds transformer with option to have encoder decoder or decoder only.
     *
     * @param neuralNetworkConfiguration neural network configuration.
     * @param decoderInputSize           decoder output size.
     * @param numberOfDecoderInputs      number of decoder inputs.
     * @param feedforwardWidth           feedforward layer width.
     * @param numberOfAttentionBlocks    number of attention blocks.
     * @return transformer layer id.
     * @throws DynamicParamException  throws exception if setting of neural network parameters fail.
     * @throws NeuralNetworkException throws exception if creation of neural network instance fails.
     * @throws MatrixException        throws exception if custom function is attempted to be created with this constructor.
     */
    public static int buildStackedAttention(NeuralNetworkConfiguration  neuralNetworkConfiguration, int decoderInputSize, int numberOfDecoderInputs, int feedforwardWidth, int numberOfAttentionBlocks) throws MatrixException, NeuralNetworkException, DynamicParamException {
        if (numberOfDecoderInputs < 2) throw new NeuralNetworkException("Stacked attention must have at least two inputs.");

        int numberOfInputBlocks = numberOfDecoderInputs / 2;
        int numberOfAdditionalInputs = numberOfDecoderInputs % 2;

        Stack<Integer> attentionIndices = new Stack<>();
        for (int inputBlockIndex = 0; inputBlockIndex < numberOfInputBlocks; inputBlockIndex++) {
            int decoderLayerIndex = buildDecoderInputAttentionBlock(neuralNetworkConfiguration, decoderInputSize, 2 + numberOfAdditionalInputs);
            attentionIndices.push(decoderLayerIndex);
            numberOfAdditionalInputs = 0;
        }

        Stack<Integer> nextAttentionIndices = new Stack<>();
        int outputAttentionIndex = -1;
        while (!attentionIndices.empty()) {
            int firstAttentionIndex = attentionIndices.pop();
            if (attentionIndices.empty()) {
                if (!nextAttentionIndices.empty()) {
                    nextAttentionIndices.push(firstAttentionIndex);
                    attentionIndices = nextAttentionIndices;
                }
                else {
                    outputAttentionIndex = firstAttentionIndex;
                }
            }
            else {
                int secondAttentionIndex = attentionIndices.pop();
                outputAttentionIndex = buildAttentionModule(neuralNetworkConfiguration, firstAttentionIndex, secondAttentionIndex);
                nextAttentionIndices.push(outputAttentionIndex);
            }
        }

        // Attention blocks
        for (int index = 0; index < numberOfAttentionBlocks; index++) {
            outputAttentionIndex = buildAttentionBlock(neuralNetworkConfiguration, outputAttentionIndex, feedforwardWidth);
        }

        return outputAttentionIndex;
    }

}
