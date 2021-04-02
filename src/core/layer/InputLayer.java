/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.layer;

import core.NeuralNetworkException;
import core.normalization.NormalizationType;
import core.optimization.Optimizer;
import core.regularization.RegularizationType;
import utils.*;
import utils.matrix.Matrix;

import java.util.HashMap;

/**
 * Defines class for input layer of neural network.
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
     * Sets if recurrent inputs of layer are allowed to be reset during training.
     *
     * @param resetStateTraining if true allows reset.
     */
    public void resetStateTraining(boolean resetStateTraining) {
    }

    /**
     * Sets if recurrent inputs of layer are allowed to be reset during testing.
     *
     * @param resetStateTesting if true allows reset.
     */
    public void resetStateTesting(boolean resetStateTesting) {
    }

    /**
     * Sets if recurrent inputs of layer are allowed to be restored during training.
     *
     * @param restoreStateTraining if true allows restore.
     */
    public void restoreStateTraining(boolean restoreStateTraining) {
    }

    /**
     * Sets if recurrent inputs of layer are allowed to be restored during testing.
     *
     * @param restoreStateTesting if true allows restore.
     */
    public void restoreStateTesting(boolean restoreStateTesting) {
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
     * Executes backward processing step. Not relevant for input layer.
     *
     */
    public void backwardProcess() {
    }

    /**
     * Executes weight updates with regularizers and optimizer.
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
     * Resets normalizers and optimizer of layer.
     *
     */
    public void reset() {
    }

    /**
     * Adds regularization method for layer.
     *
     * @param regularizationType regularization method.
     */
    public void addRegularization(RegularizationType regularizationType) {
    }

    /**
     * Adds regularization method for layer.
     *
     * @param regularizationType regularization method.
     * @param params parameters for regularizer.
     */
    public void addRegularization(RegularizationType regularizationType, String params) {
    }

    /**
     * Removes any regularization from layer.
     *
     */
    public void removeRegularization() {
    }

    /**
     * Removes specific regularization from layer.
     *
     * @param regularizationType regularization method to be removed.
     */
    public void removeRegularization(RegularizationType regularizationType) {
    }

    /**
     * Adds normalization method for layer.
     *
     * @param normalizationType normalization method.
     */
    public void addNormalization(NormalizationType normalizationType) {
    }

    /**
     * Adds normalization method for layer.
     *
     * @param normalizationType normalization method.
     * @param params parameters for normalizer.
     */
    public void addNormalization(NormalizationType normalizationType, String params) {
    }

    /**
     * Removes any normalization from layer.
     *
     */
    public void removeNormalization() {
    }

    /**
     * Removes specific normalization from layer.
     *
     * @param normalizationType normalization method to be removed.
     */
    public void removeNormalization(NormalizationType normalizationType) {
    }

    /**
     * Resets specific normalization for layer.
     *
     * @param normalizationType normalization method to be reset.
     */
    public void resetNormalization(NormalizationType normalizationType) {
    }

    /**
     * Resets all normalization for layer.
     *
     */
    public void resetNormalization() {
    }

    /**
     * Reinitializes specific normalization for layer.
     *
     * @param normalizationType normalization method to be reinitialized.
     */
    public void reinitializeNormalization(NormalizationType normalizationType) {
    }

    /**
     * Resets all normalization for layer.
     *
     */
    public void reinitializeNormalization() {
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
     * Returns ordered map of weights.
     *
     * @return ordered map of weights.
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
     * Prints expression chains of neural network.
     *
     */
    public void printExpressions() {
    }

    /**
     * Prints gradient chains of neural network.
     *
     */
    public void printGradients() {
    }

}

