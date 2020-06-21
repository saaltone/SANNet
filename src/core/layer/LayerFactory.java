/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.layer;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.convolutional.*;
import core.layer.feedforward.*;
import core.layer.recurrent.*;
import utils.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.MatrixException;

/**
 * Factory class to construct neural network layers.
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
        switch (layerType) {
            case FEEDFORWARD:
                return new FeedforwardLayer(layerIndex, activationFunction, initialization, params);
            case RECURRENT:
                return new RecurrentLayer(layerIndex, activationFunction, initialization, params);
            case LSTM:
                return new LSTMLayer(layerIndex, initialization, params);
            case PEEPHOLELSTM:
                return new PeepholeLSTMLayer(layerIndex, initialization, params);
            case GRAVESLSTM:
                return new GravesLSTMLayer(layerIndex, initialization, params);
            case GRU:
                return new GRULayer(layerIndex, initialization, params);
            case MINGRU:
                return new MinGRULayer(layerIndex, initialization, params);
            case CONVOLUTIONAL:
                return new ConvolutionalLayer(layerIndex, activationFunction, initialization, params);
            case POOLING:
                return new PoolingLayer(layerIndex, initialization, params);
        }
        return null;
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
        if (neuralNetworkLayer instanceof ConvolutionalLayer) return LayerType.CONVOLUTIONAL;
        if (neuralNetworkLayer instanceof PoolingLayer) return LayerType.POOLING;
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
        if (neuralNetworkLayer instanceof ConvolutionalLayer) return "CONVOLUTIONAL";
        if (neuralNetworkLayer instanceof PoolingLayer) return "POOLING";
        throw new NeuralNetworkException("Unknown layer type");
    }

}
