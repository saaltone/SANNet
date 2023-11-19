/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer;

import core.loss.LossFunction;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.operation.BinaryMatrixOperation;
import utils.sampling.Sequence;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Implements output layer of neural network.<br>
 * Outputs inference result of neural network.<br>
 * Calculates loss and its gradient during training phase.<br>
 *
 */
public class OutputLayer extends AbstractPlainLayer {

    /**
     * Layer group index.
     *
     */
    private final int layerGroupIndex;

    /**
     * Neural network loss function for output layer (single output case).
     *
     */
    private final LossFunction lossFunction;

    /**
     * Neural network output error.
     *
     */
    private transient Matrix loss;

    /**
     * Target (actual true) output values for error calculation in training phase.
     *
     */
    private transient Sequence targets;

    /**
     * Importance sampling weights for gradient calculation.
     *
     */
    private transient HashMap<Integer, Double> importanceSamplingWeights;

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
        this(layerIndex, -1,  lossFunction);
    }

    /**
     * Constructor for output layer.
     *
     * @param layerIndex index of layer.
     * @param layerGroupIndex index of layer group.
     * @param lossFunction loss function for output layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public OutputLayer(int layerIndex, int layerGroupIndex, LossFunction lossFunction) throws NeuralNetworkException, DynamicParamException {
        super(layerIndex, null);
        this.lossFunction = lossFunction;
        this.layerGroupIndex = layerGroupIndex > -1 ? layerGroupIndex : 0;
    }

    /**
     * Sets reference to next neural network layer.
     *
     * @param nextLayer reference to next neural network layer.
     * @throws NeuralNetworkException throws exception if next layer is attempted to be added to output layer.
     */
    public void addNextLayer(NeuralNetworkLayer nextLayer) throws NeuralNetworkException {
        throw new NeuralNetworkException("Output layer cannot have next layers.");
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
     * Sets reset flag for procedure expression dependencies.
     *
     * @param resetDependencies if true procedure expression dependencies are reset otherwise false.
     */
    public void resetDependencies(boolean resetDependencies) {
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
        return getDefaultLayerInput();
    }

    /**
     * Executes forward processing step of output layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardProcess() throws MatrixException {
        if (targets == null || targets.isEmpty() || !training) return;
        loss = null;
        for (Map.Entry<Integer, Matrix> entry : targets.entrySet()) {
            int sampleIndex = entry.getKey();
            Matrix output = getLayerOutputs().get(sampleIndex);
            Matrix target = entry.getValue();
            Matrix currentLoss = new BinaryMatrixOperation(output.getRows(), output.getColumns(), output.getDepth(), lossFunction).applyFunction(output, target);
            if (importanceSamplingWeights != null) currentLoss.multiplyBy(importanceSamplingWeights.get(sampleIndex));
            loss = loss == null ? currentLoss : loss.add(currentLoss);
        }
        loss = LossFunction.getMeanError(loss, targets.totalSize());
    }

    /**
     * Executes backward step of neural network.
     *
     * @throws NeuralNetworkException throws exception if targets are not set or output and target dimensions are not matching.
     */
    public void backward() throws NeuralNetworkException  {
        if (targets.isEmpty()) throw new NeuralNetworkException("No targets defined");
        if (targets.totalSize() != getLayerOutputs().totalSize()) throw new NeuralNetworkException("Target size: "+ targets.totalSize() + " is not matching with output size: " + getLayerOutputs().totalSize());
        super.backward(true);
    }

    /**
     * Executes backward step of output layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardProcess() throws MatrixException {
        Sequence lossGradients = new Sequence();
        for (Map.Entry<Integer, Matrix> entry : targets.entrySet()) {
            int sampleIndex = entry.getKey();
            Matrix output = getLayerOutputs().get(sampleIndex);
            Matrix target = entry.getValue();
            Matrix currentLossGradient = new BinaryMatrixOperation(output.getRows(), output.getColumns(), output.getDepth(), lossFunction).applyGradient(output, target);
            if (importanceSamplingWeights != null) currentLossGradient.multiplyBy(importanceSamplingWeights.get(sampleIndex));
            lossGradients.put(sampleIndex, currentLossGradient);
        }
        getDefaultLayerInputGradient().increment(lossGradients);
    }

    /**
     * Sets importance sampling weights.
     *
     * @param importanceSamplingWeights importance sampling weights.
     */
    public void setImportanceSamplingWeights(HashMap<Integer, Double> importanceSamplingWeights) {
        this.importanceSamplingWeights = importanceSamplingWeights;
    }

    /**
     * Cumulates error from regularization. Mainly from L1 / L2 / Lp regularization.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return cumulated error from regularization.
     */
    public double error() throws MatrixException, DynamicParamException {
        return hasPreviousLayers() ? getDefaultPreviousLayer().error() / (double)targets.totalSize() : 0;
    }

    /**
     * Returns total error of neural network including impact of regularization.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return total error of neural network.
     */
    public double getTotalError() throws MatrixException, DynamicParamException {
        return (loss == null || targets == null) ? 0 : lossFunction.getAbsoluteError(loss) + error();
    }

    /**
     * Prints structure and metadata of neural network layer.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void print() throws NeuralNetworkException {
        System.out.println(getLayerName() + " [ " + getLayerConnections() + ", Loss function: " + lossFunction.getName() + ", Layer Group ID: " + layerGroupIndex + " ]");
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerConnections() {
        ArrayList<Integer> inputLayerList = new ArrayList<>();
        for (NeuralNetworkLayer previousLayer : getPreviousLayers().values()) inputLayerList.add(previousLayer.getLayerIndex());
        return "Connect from layers: " + (!inputLayerList.isEmpty() ? inputLayerList : "N/A");
    }

}