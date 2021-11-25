/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer;

import core.network.NeuralNetworkException;
import core.optimization.Optimizer;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;

import java.util.HashMap;
import java.util.HashSet;

/**
 * Defines class for input layer of neural network.<br>
 *
 */
public class InputLayer extends AbstractLayer {

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
     * Checks if execution layer is convolutional layer type.
     *
     * @return true if execution layer is convolutional layer type otherwise false.
     */
    public boolean isConvolutionalLayer() {
        return false;
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     */
    protected void defineProcedure() {
    }

    /**
     * Initializes layer.
     *
     */
    public void initialize() {
    }

    /**
     * Reinitializes layer.
     *
     */
    public void reinitialize() {
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
     * Cumulates error from regularization. Mainly from L1 / L2 / Lp regularization.
     *
     * @return cumulated error from regularization.
     */
    public double error() {
        return 0;
    }

    /**
     * Resets optimizer of layer.
     *
     */
    public void reset() {
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
     * Prints structure and metadata of neural network.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void print() throws NeuralNetworkException {
        System.out.println(getLayerName() + " [ Width: " + getLayerWidth() + ", Height: " + getLayerHeight() + ", Depth: " + getLayerDepth() + " ]");
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

