/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer;

import core.layer.normalization.BatchNormalization;
import core.layer.normalization.LayerNormalization;
import core.layer.normalization.WeightNormalization;
import core.layer.regularization.*;
import core.network.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.convolutional.*;
import core.layer.feedforward.*;
import core.layer.recurrent.*;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.MatrixException;

/**
 * Factory class to construct neural network layers.<br>
 *
 */
public class LayerFactory {

    /**
     * Creates neural network execution layer.
     *
     * @param layerIndex layer Index.
     * @param layerType type of layer.
     * @param activationFunction activation function for layer.
     * @param initialization initialization function for layer.
     * @param params parameters for layer.
     * @return created neural network layer instance
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public static AbstractExecutionLayer create(int layerIndex, LayerType layerType, ActivationFunction activationFunction, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException, MatrixException {
        return switch (layerType) {
            case FEEDFORWARD -> new FeedforwardLayer(layerIndex, activationFunction, initialization, params);
            case RECURRENT -> new RecurrentLayer(layerIndex, activationFunction, initialization, params);
            case LSTM -> new LSTMLayer(layerIndex, activationFunction, initialization, params);
            case PEEPHOLELSTM -> new PeepholeLSTMLayer(layerIndex, activationFunction, initialization, params);
            case GRAVESLSTM -> new GravesLSTMLayer(layerIndex, activationFunction, initialization, params);
            case GRU -> new GRULayer(layerIndex, initialization, params);
            case MINGRU -> new MinGRULayer(layerIndex, initialization, params);
            case CONVOLUTION -> new ConvolutionLayer(layerIndex, activationFunction, initialization, params);
            case CROSSCORRELATION -> new CrosscorrelationLayer(layerIndex, activationFunction, initialization, params);
            case WINOGRAD_CONVOLUTION -> new WinogradConvolutionLayer(layerIndex, activationFunction, initialization, params);
            case MAX_POOLING -> new MaxPoolingLayer(layerIndex, initialization, params);
            case RANDOM_POOLING -> new RandomPoolingLayer(layerIndex, initialization, params);
            case CYCLIC_POOLING -> new CyclicPoolingLayer(layerIndex, initialization, params);
            case AVERAGE_POOLING -> new AveragePoolingLayer(layerIndex, initialization, params);
            case BATCH_NORMALIZATION -> new BatchNormalization(layerIndex, initialization, params);
            case LAYER_NORMALIZATION -> new LayerNormalization(layerIndex, initialization, params);
            case WEIGHT_NORMALIZATION -> new WeightNormalization(layerIndex, initialization, params);
            case DROPOUT -> new Dropout(layerIndex, initialization, params);
            case GRADIENT_CLIPPING -> new GradientClipping(layerIndex, initialization, params);
            case L1_REGULARIZATION -> new L1_Regularization(layerIndex, initialization, params);
            case L2_REGULARIZATION -> new L2_Regularization(layerIndex, initialization, params);
            case Lp_REGULARIZATION -> new Lp_Regularization(layerIndex, initialization, params);
            case WEIGHT_NOISING -> new WeightNoising(layerIndex, initialization, params);
        };
    }

    /**
     * Returns type of a given layer.
     *
     * @param neuralNetworkLayer given layer.
     * @return type of a layer.
     * @throws NeuralNetworkException throws exception if layer is of an unknown type.
     */
    public static LayerType getLayerType(NeuralNetworkLayer neuralNetworkLayer) throws NeuralNetworkException {
        if (neuralNetworkLayer instanceof FeedforwardLayer) return LayerType.FEEDFORWARD;
        if (neuralNetworkLayer instanceof RecurrentLayer) return LayerType.RECURRENT;
        if (neuralNetworkLayer instanceof LSTMLayer) return LayerType.LSTM;
        if (neuralNetworkLayer instanceof PeepholeLSTMLayer) return LayerType.PEEPHOLELSTM;
        if (neuralNetworkLayer instanceof GravesLSTMLayer) return LayerType.GRAVESLSTM;
        if (neuralNetworkLayer instanceof GRULayer) return LayerType.GRU;
        if (neuralNetworkLayer instanceof MinGRULayer) return LayerType.MINGRU;
        if (neuralNetworkLayer instanceof ConvolutionLayer) return LayerType.CONVOLUTION;
        if (neuralNetworkLayer instanceof CrosscorrelationLayer) return LayerType.CROSSCORRELATION;
        if (neuralNetworkLayer instanceof WinogradConvolutionLayer) return LayerType.WINOGRAD_CONVOLUTION;
        if (neuralNetworkLayer instanceof MaxPoolingLayer) return LayerType.MAX_POOLING;
        if (neuralNetworkLayer instanceof RandomPoolingLayer) return LayerType.RANDOM_POOLING;
        if (neuralNetworkLayer instanceof CyclicPoolingLayer) return LayerType.CYCLIC_POOLING;
        if (neuralNetworkLayer instanceof AveragePoolingLayer) return LayerType.AVERAGE_POOLING;
        if (neuralNetworkLayer instanceof BatchNormalization) return LayerType.BATCH_NORMALIZATION;
        if (neuralNetworkLayer instanceof LayerNormalization) return LayerType.LAYER_NORMALIZATION;
        if (neuralNetworkLayer instanceof WeightNormalization) return LayerType.WEIGHT_NORMALIZATION;
        if (neuralNetworkLayer instanceof Dropout) return LayerType.DROPOUT;
        if (neuralNetworkLayer instanceof GradientClipping) return LayerType.GRADIENT_CLIPPING;
        if (neuralNetworkLayer instanceof L1_Regularization) return LayerType.L1_REGULARIZATION;
        if (neuralNetworkLayer instanceof L2_Regularization) return LayerType.L2_REGULARIZATION;
        if (neuralNetworkLayer instanceof Lp_Regularization) return LayerType.Lp_REGULARIZATION;
        if (neuralNetworkLayer instanceof WeightNoising) return LayerType.WEIGHT_NOISING;
        throw new NeuralNetworkException("Unknown layer type");
    }

    /**
     * Returns type of layer as string.
     *
     * @param neuralNetworkLayer given layer.
     * @return type of a layer as string.
     * @throws NeuralNetworkException throws exception if layer is of an unknown type.
     */
    public static String getLayerTypeByName(NeuralNetworkLayer neuralNetworkLayer) throws NeuralNetworkException {
        if (neuralNetworkLayer instanceof FeedforwardLayer) return "FEEDFORWARD";
        if (neuralNetworkLayer instanceof RecurrentLayer) return "RECURRENT";
        if (neuralNetworkLayer instanceof LSTMLayer) return "LSTM";
        if (neuralNetworkLayer instanceof PeepholeLSTMLayer) return "PEEPHOLELSTM";
        if (neuralNetworkLayer instanceof GravesLSTMLayer) return "GRAVESLSTM";
        if (neuralNetworkLayer instanceof GRULayer) return "GRU";
        if (neuralNetworkLayer instanceof MinGRULayer) return "MINGRU";
        if (neuralNetworkLayer instanceof ConvolutionLayer) return "CONVOLUTION";
        if (neuralNetworkLayer instanceof CrosscorrelationLayer) return "CROSSCORRELATION";
        if (neuralNetworkLayer instanceof WinogradConvolutionLayer) return "WINOGRAD_CONVOLUTION";
        if (neuralNetworkLayer instanceof MaxPoolingLayer) return "MAX_POOLING";
        if (neuralNetworkLayer instanceof RandomPoolingLayer) return "RANDOM_POOLING";
        if (neuralNetworkLayer instanceof CyclicPoolingLayer) return "CYCLIC_POOLING";
        if (neuralNetworkLayer instanceof AveragePoolingLayer) return "AVERAGE_POOLING";
        if (neuralNetworkLayer instanceof BatchNormalization) return "BATCH_NORMALIZATION";
        if (neuralNetworkLayer instanceof LayerNormalization) return "LAYER_NORMALIZATION";
        if (neuralNetworkLayer instanceof WeightNormalization) return "WEIGHT_NORMALIZATION";
        if (neuralNetworkLayer instanceof Dropout) return "DROPOUT";
        if (neuralNetworkLayer instanceof GradientClipping) return "GRADIENT_CLIPPING";
        if (neuralNetworkLayer instanceof L1_Regularization) return "L1_REGULARIZATION";
        if (neuralNetworkLayer instanceof L2_Regularization) return "L2_REGULARIZATION";
        if (neuralNetworkLayer instanceof Lp_Regularization) return "Lp_REGULARIZATION";
        if (neuralNetworkLayer instanceof WeightNoising) return "WEIGHT_NOISING";
        throw new NeuralNetworkException("Unknown layer type");
    }

}
