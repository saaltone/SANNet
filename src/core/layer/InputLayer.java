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
     * Layer group index.
     *
     */
    private final int layerGroupIndex;

    /**
     * Constructor for input layer.
     *
     * @param layerIndex index of layer.
     * @param params parameters for input layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public InputLayer(int layerIndex, String params) throws DynamicParamException, NeuralNetworkException {
        this(layerIndex, -1, params);
    }

    /**
     * Constructor for input layer.
     *
     * @param layerIndex index of layer.
     * @param layerGroupIndex index of layer group.
     * @param params parameters for input layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public InputLayer(int layerIndex, int layerGroupIndex, String params) throws DynamicParamException, NeuralNetworkException {
        super(layerIndex, params);
        this.layerGroupIndex = layerGroupIndex > -1 ? layerGroupIndex : 0;
    }

    /**
     * Initializes neural network layer dimensions.
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    public void initializeDimensions() throws NeuralNetworkException {
        if (getLayerWidth() < 1) throw new NeuralNetworkException("Input layer width must be positive. Invalid value: " + getLayerWidth());
        if (getLayerHeight() < 1) throw new NeuralNetworkException("Input height width must be positive. Invalid value: " + getLayerHeight());
        if (getLayerDepth() < 1) throw new NeuralNetworkException("Input depth width must be positive. Invalid value: " + getLayerDepth());
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
        System.out.println(getLayerName() + " [ Width: " + getLayerWidth() + ", Height: " + getLayerHeight() + ", Depth: " + getLayerDepth() + ", Layer Group ID: " + layerGroupIndex + " ]");
    }

}

