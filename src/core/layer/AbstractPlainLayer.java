/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer;

import core.network.NeuralNetworkException;
import core.optimization.Optimizer;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;

import java.util.HashMap;
import java.util.HashSet;

/**
 * Implements abstract plain layer.<br>
 * Provides common functions for input and output layers.<br>
 *
 */
public abstract class AbstractPlainLayer extends AbstractLayer {

    /**
     * Constructor for abstract plain layer.
     *
     * @param layerIndex index of layer.
     * @param params parameters for input layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public AbstractPlainLayer(int layerIndex, String params) throws DynamicParamException, NeuralNetworkException {
        super(layerIndex, params);
    }

    /**
     * Returns layer type by name
     *
     * @return layer type by name
     */
    protected String getTypeByName() {
        return "";
    }

    /**
     * Checks if execution layer is recurrent layer type.
     *
     * @return true if execution layer is recurrent layer type otherwise false.
     */
    public boolean isRecurrentLayer() {
        return false;
    }

    /**
     * Checks if layer works with recurrent layers.
     *
     * @return if true layer works with recurrent layers otherwise false.
     */
    public boolean worksWithRecurrentLayer() {
        return true;
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     */
    protected void defineProcedure() {
    }

    /**
     * Reinitializes layer.
     *
     */
    public void reinitialize() {
    }

    /**
     * Returns weights for normalization.
     *
     * @return weights for normalization.
     */
    public HashSet<Matrix> getNormalizedWeights() {
        return null;
    }

    /**
     * Returns weights for regularization.
     *
     * @return weights for regularization.
     */
    public HashSet<Matrix> getRegularizedWeights() {
        return null;
    }

    /**
     * Returns neural network weight gradients.
     *
     * @return neural network weight gradients.
     */
    public HashMap<Matrix, Matrix> getLayerWeightGradients() {
        return null;
    }

    /**
     * Executes weight updates with optimizer.
     *
     */
    public void optimize() {
    }

    /**
     * Sets optimizer for layer.<br>
     * Optimizer optimizes weight parameters iteratively towards optimal solution.<br>
     *
     * @param optimizer optimizer to be added.
     */
    public void setOptimizer(Optimizer optimizer) {
    }

    /**
     * Resets optimizer for layer.
     *
     */
    public void resetOptimizer() {
    }

    /**
     * Returns map of weights.
     *
     * @return map of weights.
     */
    public HashMap<Integer, Matrix> getWeightsMap() {
        return null;
    }

    /**
     * Appends other neural network layer with equal weights to this layer by weighted factor tau.
     *
     * @param otherNeuralNetworkLayer other neural network layer.
     * @param tau tau which controls contribution of other layer.
     */
    public void append(NeuralNetworkLayer otherNeuralNetworkLayer, double tau) {
    }

    /**
     * Returns number of layer parameters.
     *
     * @return number of layer parameters.
     */
    public int getNumberOfParameters() {
        return 0;
    }

    /**
     * Prints expression chains of neural network layer.
     *
     */
    public void printExpressions() {
    }

    /**
     * Prints gradient chains of neural network layer.
     *
     */
    public void printGradients() {
    }

}
