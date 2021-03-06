/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.layer;

import core.NeuralNetworkException;
import core.loss.LossFunction;
import core.normalization.NormalizationType;
import core.optimization.Optimizer;
import core.regularization.RegularizationType;
import utils.*;
import utils.matrix.*;

import java.util.HashMap;
import java.util.TreeMap;

/**
 * Defines class for output layer of neural network.<br>
 *
 */
public class OutputLayer extends AbstractLayer {

    /**
     * Neural network loss function for output layer.
     *
     */
    private final LossFunction lossFunction;

    /**
     * Neural network output error.
     *
     */
    private transient Matrix error;

    /**
     * Target (actual true) output values for error calculation in training phase.
     *
     */
    private transient Sequence targets;

    /**
     * Importance sampling weights for gradient calculation.
     *
     */
    private transient TreeMap<Integer, Double> importanceSamplingWeights;

    /**
     * If true neural network is in training state otherwise false.
     *
     */
    private transient boolean training;

    /**
     * Constructor for output layer.
     *
     * @param layerIndex index of layer.
     * @param lossFunction loss function for output layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public OutputLayer(int layerIndex, LossFunction lossFunction) throws NeuralNetworkException, DynamicParamException {
        super(layerIndex, null);
        this.lossFunction = lossFunction;
    }

    /**
     * Returns loss function type.
     *
     * @return loss function type.
     */
    public BinaryFunctionType getLossFunctionType() {
        return lossFunction.getType();
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
        this.training = training;
    }

    /**
     * Sets targets (actual true output values) of neural network (output layer).<br>
     * In error calculation predicted output is compared to actual true output values.<br>
     *
     * @param targets targets of output layer.
     */
    public void setTargets(Sequence targets) {
        this.targets = targets;
    }

    /**
     * Returns outputs of neural network layer.
     *
     * @return outputs of neural network layer.
     */
    public Sequence getLayerOutputs() {
        return getPreviousLayer().getLayerOutputs();
    }

    /**
     * Sets importance sampling weights.
     *
     * @param importanceSamplingWeights importance sampling weights.
     */
    public void setImportanceSamplingWeights(TreeMap<Integer, Double> importanceSamplingWeights) {
        this.importanceSamplingWeights = importanceSamplingWeights;
    }

    /**
     * Executes backward step of neural network.
     *
     * @throws NeuralNetworkException throws exception if targets are not set or output and target dimensions are not matching.
     */
    public void backward() throws NeuralNetworkException  {
        if (targets.isEmpty()) throw new NeuralNetworkException("No targets defined");
        if (targets.totalSize() != getLayerOutputs().totalSize()) throw new NeuralNetworkException("Target size: "+ targets.totalSize() + " is not matching with output size: " + getLayerOutputs().totalSize());
        super.backward();
    }

    /**
     * Executes forward processing step of output layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardProcess() throws MatrixException {
        if (targets == null || targets.isEmpty() || !training) return;
        error = null;
        for (Integer sampleIndex : targets.keySet()) {
            for (Integer matrixIndex : targets.sampleKeySet()) {
                Matrix outputError = lossFunction.getError(getLayerOutputs().get(sampleIndex, matrixIndex), targets.get(sampleIndex, matrixIndex));
                if (importanceSamplingWeights != null) outputError.multiply(importanceSamplingWeights.get(matrixIndex), outputError);
                error = error == null ? outputError : error.add(outputError);
            }
        }
        error = lossFunction.getMeanError(error, targets.totalSize());
    }

    /**
     * Executes backward step of output layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardProcess() throws MatrixException {
        resetLayerGradients();
        for (Integer sampleIndex : targets.keySet()) {
            for (Integer matrixIndex : targets.sampleKeySet()) {
                Matrix outputGradient = lossFunction.getGradient(getLayerOutputs().get(sampleIndex, matrixIndex), targets.get(sampleIndex, matrixIndex));
                if (importanceSamplingWeights != null) outputGradient.multiply(importanceSamplingWeights.get(matrixIndex), outputGradient);
                getLayerGradients().put(sampleIndex, matrixIndex, outputGradient);
            }
        }
    }

    /**
     * Returns total error of neural network including impact of regularization.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return total error of neural network.
     */
    public double getTotalError() throws MatrixException, DynamicParamException {
        return (error == null || targets == null) ? 0 : error.mean() + error();
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
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return cumulated error from regularization.
     */
    public double error() throws MatrixException, DynamicParamException {
        return hasPreviousLayer() ? getPreviousLayer().error() / (double)targets.totalSize() : 0;
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
     * Prints structure and metadata of neural network layer.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void print() throws NeuralNetworkException {
        System.out.println(getLayerName() + " [ Loss function: " + lossFunction.getName() + " ]");
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