/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer;

import core.network.NeuralNetworkException;
import core.loss.LossFunction;
import core.optimization.Optimizer;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.sampling.Sequence;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeMap;

/**
 * Defines class for output layer of neural network.<br>
 *
 */
public class OutputLayer extends AbstractLayer {

    /**
     * Neural network loss function for output layer (single output case).
     *
     */
    private final LossFunction lossFunction;

    /**
     * Neural network loss functions for output layer (multi-output case).
     *
     */
    private final ArrayList<LossFunction> lossFunctions = new ArrayList<>();

    /**
     * If true output layer has multiple outputs otherwise single output.
     *
     */
    private final boolean multiOutput;

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
        this.multiOutput = false;
    }

    /**
     * Constructor for output layer.
     *
     * @param layerIndex index of layer.
     * @param lossFunctions loss functions for output layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public OutputLayer(int layerIndex, ArrayList<LossFunction> lossFunctions) throws NeuralNetworkException, DynamicParamException {
        super(layerIndex, null);
        this.lossFunction = null;
        this.lossFunctions.addAll(lossFunctions);
        this.multiOutput = true;
    }

    /**
     * Returns if output layer has multiple outputs.
     *
     * @return if true output layer has multiple outputs otherwise single output.
     */
    public boolean isMultiOutput() {
        return multiOutput;
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
     * Executes forward processing step of output layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardProcess() throws MatrixException {
        if (targets == null || targets.isEmpty() || !training) return;
        error = null;
        for (Integer sampleIndex : targets.keySet()) {
            for (Integer matrixIndex : targets.sampleKeySet()) {
                Matrix currentOutputs = getLayerOutputs().get(sampleIndex, matrixIndex);
                Matrix currentTargets = targets.get(sampleIndex, matrixIndex);
                Matrix outputError;
                if (multiOutput) {
                    if (currentOutputs.getSubMatrices().size() != lossFunctions.size()) throw new MatrixException("Number of outputs is not matching with number of loss functions");
                    ArrayList<Matrix> totalError = new ArrayList<>();
                    int lossFunctionsSize = lossFunctions.size();
                    for (int index = 0; index < lossFunctionsSize; index++) {
                        LossFunction lossFunction = lossFunctions.get(index);
                        Matrix subOutputs = currentOutputs.getSubMatrices().get(index);
                        Matrix subTargets = currentTargets.getSubMatrices().get(index);
                        Matrix subOutputError = lossFunction.getError(subOutputs, subTargets);
                        totalError.add(subOutputError);
                    }
                    outputError = new JMatrix(currentOutputs.getTotalRows(), currentOutputs.getTotalColumns(), totalError, true);
                }
                else {
                    outputError = lossFunction.getError(currentOutputs, currentTargets);
                }
                error = error == null ? outputError : error.add(outputError);
            }
        }
        error = LossFunction.getMeanError(error, targets.totalSize());
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
                Matrix currentOutputs = getLayerOutputs().get(sampleIndex, matrixIndex);
                Matrix currentTargets = targets.get(sampleIndex, matrixIndex);
                Matrix outputGradient;
                if (multiOutput) {
                    if (currentOutputs.getSubMatrices().size() != lossFunctions.size()) throw new MatrixException("Number of outputs is not matching with number of loss functions");
                    ArrayList<Matrix> totalGradient = new ArrayList<>();
                    int lossFunctionsSize = lossFunctions.size();
                    for (int index = 0; index < lossFunctionsSize; index++) {
                        LossFunction lossFunction = lossFunctions.get(index);
                        Matrix subOutputs = currentOutputs.getSubMatrices().get(index);
                        Matrix subTargets = currentTargets.getSubMatrices().get(index);
                        Matrix subOutputGradient = lossFunction.getGradient(subOutputs, subTargets);
                        totalGradient.add(subOutputGradient);
                    }
                    outputGradient = new JMatrix(currentOutputs.getTotalRows(), currentOutputs.getTotalColumns(), totalGradient, true);
                }
                else {
                    outputGradient = lossFunction.getGradient(currentOutputs, currentTargets);
                }
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
        if (!multiOutput) System.out.println(getLayerName() + " [ Loss function: " + lossFunction.getName() + " ]");
        else {
            System.out.print(getLayerName() + " [ Loss functions: ");
            for (LossFunction lossFunction : lossFunctions) System.out.print(lossFunction.getName() + " ");
            System.out.println("]");
        }
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