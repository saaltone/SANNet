/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;

/**
 * Defines class for input layer of neural network.<br>
 *
 */
public class InputLayer extends AbstractPlainLayer {

    /**
     * Constructor for input layer.
     *
     * @param layerIndex index of layer.
     * @param params parameters for input layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public InputLayer(int layerIndex, String params) throws DynamicParamException, NeuralNetworkException {
        super(layerIndex, params);
    }

    /**
     * Executes parameter (weight) update for training step of neural network layer.
     *
     */
    public void update(){
        super.update();
        waitToComplete();
    }

    /**
     * Sets training flag.
     *
     * @param training if true layer is training otherwise false.
     */
    protected void setTraining(boolean training) {
    }

    /**
     * Executes forward processing step of execution layer.
     *
     */
    public void forwardProcess() {
    }

    /**
     * Executes backward processing step.
     *
     */
    public void backwardProcess() {
    }

    /**
     * Cumulates error from regularization. Mainly from L1 / L2 / Lp regularization.
     *
     * @return cumulated error from regularization.
     */
    public double error() {
        return 0;
    }

    /**
     * Prints structure and metadata of neural network.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void print() throws NeuralNetworkException {
        System.out.println(getLayerName() + " [ Width: " + getLayerWidth() + ", Height: " + getLayerHeight() + ", Depth: " + getLayerDepth() + " ]");
    }

}

