/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;

/**
 * Implements input layer of neural network.<br>
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
     /**
     * Sets reference to previous neural network layer.
     *
     * @param previousLayer reference to previous neural network layer.
     * @throws NeuralNetworkException throws exception if previous layer is attempted to be added to input layer.
     */
    public void addPreviousLayer(NeuralNetworkLayer previousLayer) throws NeuralNetworkException {
        throw new NeuralNetworkException("Input layer cannot have previous layers.");
    }

    /**
     * Sets training flag.
     *
     * @param training if true layer is training otherwise false.
     */
    protected void setTraining(boolean training) {
    }

    /**
     * Sets reset flag for procedure expression dependencies.
     *
     * @param resetDependencies if true procedure expression dependencies are reset otherwise false.
     */
    public void resetDependencies(boolean resetDependencies) {
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

