/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.layer;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.convolutional.ConvolutionalLayer;
import core.layer.convolutional.PoolingLayer;
import core.layer.feedforward.FeedforwardLayer;
import core.layer.recurrent.GRULayer;
import core.layer.recurrent.GravesLSTMLayer;
import core.layer.recurrent.LSTMLayer;
import core.layer.recurrent.RecurrentLayer;
import utils.DynamicParamException;
import utils.Init;

/**
 * Factory class to construct neural network layers.
 *
 */
public class LayerFactory {

    /**
     * Creates neural network layer.
     *
     * @param layerType type of layer.
     * @param abstractLayer reference to abstract layer.
     * @param activationFunction activation function for layer.
     * @param intialization initialization function for layer.
     * @param params parameters for layer.
     * @return created neural network layer instance
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static Layer create(LayerType layerType, AbstractLayer abstractLayer, ActivationFunction activationFunction, Init intialization, String params) throws DynamicParamException {
        switch (layerType) {
            case FEEDFORWARD:
                return new FeedforwardLayer(abstractLayer, activationFunction, intialization, params);
            case RECURRENT:
                return new RecurrentLayer(abstractLayer, activationFunction, intialization, params);
            case LSTM:
                return new LSTMLayer(abstractLayer, activationFunction, intialization, params);
            case GRAVESLSTM:
                return new GravesLSTMLayer(abstractLayer, activationFunction, intialization, params);
            case GRU:
                return new GRULayer(abstractLayer, activationFunction, intialization, params);
            case CONVOLUTIONAL:
                return new ConvolutionalLayer(abstractLayer, activationFunction, intialization, params);
            case POOLING:
                return new PoolingLayer(abstractLayer, activationFunction, intialization, params);
        }
        return null;
    }

    /**
     * Returns type of a given layer.
     *
     * @param layer given layer.
     * @return type of a layer.
     * @throws NeuralNetworkException throws exception if layer is of an unknown type.
     */
    public static LayerType getLayerType(Layer layer) throws NeuralNetworkException {
        if (layer instanceof FeedforwardLayer) return LayerType.FEEDFORWARD;
        if (layer instanceof RecurrentLayer) return LayerType.RECURRENT;
        if (layer instanceof LSTMLayer) return LayerType.LSTM;
        if (layer instanceof GravesLSTMLayer) return LayerType.GRAVESLSTM;
        if (layer instanceof GRULayer) return LayerType.GRU;
        if (layer instanceof ConvolutionalLayer) return LayerType.CONVOLUTIONAL;
        if (layer instanceof PoolingLayer) return LayerType.POOLING;
        throw new NeuralNetworkException("Unknown layer type");
    }

    /**
     * Return type of layer as string.
     *
     * @param layer given layer.
     * @return type of a layer as string.
     * @throws NeuralNetworkException throws exception if layer is of an unknown type.
     */
    public static String getLayerTypeByName(Layer layer) throws NeuralNetworkException {
        if (layer instanceof FeedforwardLayer) return "FEEDFORWARD";
        if (layer instanceof RecurrentLayer) return "RECURRENT";
        if (layer instanceof LSTMLayer) return "LSTM";
        if (layer instanceof GravesLSTMLayer) return "GRAVESLSTM";
        if (layer instanceof GRULayer) return "GRU";
        if (layer instanceof ConvolutionalLayer) return "CONVOLUTIONAL";
        if (layer instanceof PoolingLayer) return "POOLING";
        throw new NeuralNetworkException("Unknown layer type");
    }

}
