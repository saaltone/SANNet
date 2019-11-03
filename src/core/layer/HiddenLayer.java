/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.layer;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import utils.DynamicParamException;
import utils.Sequence;
import utils.matrix.Init;

/**
 * Defines class for hidden layer of neural network.
 *
 */
public class HiddenLayer extends AbstractLayer {

    /**
     * Constructor for hidden layer
     *
     * @param layerIndex index of layer.
     * @param layerType defines type of layer.
     * @param activationFunction activation function for hidden layer.
     * @param initialization initialization functio for hidden layer.
     * @param params parameters for hidden layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails.
     */
    public HiddenLayer(int layerIndex, LayerType layerType, ActivationFunction activationFunction, Init initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super(layerIndex);
        super.setExecutionLayer(LayerFactory.create(layerType, this, activationFunction, initialization, params));
    }

    /**
     * Updates output error. Relevant only for output layer.
     *
     */
    public void updateOutputError() {}

    /**
     * Returns gradients of next neural network layer.
     *
     * @return gradients of next neural network layer
     */
    public Sequence getdEosN() {
        return getForward().getNLayer().getdEos();
    }

}

