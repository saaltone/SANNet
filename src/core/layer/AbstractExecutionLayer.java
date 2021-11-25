/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer;

import core.network.NeuralNetworkException;
import core.optimization.OptimizationType;
import core.optimization.Optimizer;
import core.optimization.OptimizerFactory;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.procedure.ForwardProcedure;
import utils.procedure.Procedure;
import utils.procedure.ProcedureFactory;
import utils.sampling.Sequence;

import java.util.*;

/**
 * Abstract class that implements execution layer for actual neural network layers (feed forward layer, recurrent layer etc.)<br>
 * Provides supportive functions for actual neural network layers.<br>
 * Support automatic gradient i.e. backward gradient calculation for layers supporting it.<br>
 *
 */
public abstract class AbstractExecutionLayer extends AbstractLayer implements ForwardProcedure {

    /**
     * Initialization function for neural network layer.
     *
     */
    protected Initialization initialization = Initialization.UNIFORM_XAVIER;

    /**
     * Procedure for layer. Procedure contains chain of forward and backward expressions.
     *
     */
    protected Procedure procedure = null;

    /**
     * Set of weights to be managed.
     *
     */
    private final HashSet<Matrix> weights = new HashSet<>();

    /**
     * Weights to be normalized.
     *
     */
    private final HashSet<Matrix> normalizedWeights = new HashSet<>();

    /**
     * Weights to be regularized.
     *
     */
    private final HashSet<Matrix> regularizedWeights = new HashSet<>();

    /**
     * Ordered map of weights.
     *
     */
    private final HashMap<Integer, Matrix> weightsMap = new HashMap<>();

    /**
     * Gradient sum of weights.
     *
     */
    private transient HashMap<Matrix, Matrix> weightGradientSums;

    /**
     * Optimizer for layer.
     *
     */
    private Optimizer optimizer = OptimizerFactory.create(OptimizationType.ADAM);

    /**
     * If true neural network is in training mode otherwise false.
     *
     */
    private transient boolean isTraining;

    /**
     * Constructor for abstract execution layer.
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function.
     * @param params parameters for neural network layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails.
     */
    protected AbstractExecutionLayer(int layerIndex, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super (layerIndex, params);
        if (initialization != null) this.initialization = initialization;
    }

    /**
     * Returns layer type by name.
     *
     * @return layer type by name.
     * @throws NeuralNetworkException throws exception if operation fails.
     */
    public String getTypeByName() throws NeuralNetworkException  {
        return LayerFactory.getLayerTypeByName(this);
    }

    /**
     * Initializes layer.
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void initialize() throws NeuralNetworkException, MatrixException, DynamicParamException {
        if (getLayerWidth() == -1) {
            setLayerWidth(getPreviousLayerWidth());
            setLayerHeight(getPreviousLayerHeight());
            setLayerDepth(getPreviousLayerDepth());
        }
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void defineProcedure() throws MatrixException, DynamicParamException, NeuralNetworkException {
        procedure = new ProcedureFactory().getProcedure(this, getAllConstantMatrices());
        procedure.setStopGradient(getStopGradients(), true);
    }

    /**
     * Returns all constant matrices.
     *
     * @return all constant matrices.
     */
    private HashSet<Matrix> getAllConstantMatrices() {
        HashSet<Matrix> allConstantMatrices = new HashSet<>();
        allConstantMatrices.addAll(getWeights());
        allConstantMatrices.addAll(getConstantMatrices());
        return allConstantMatrices;
    }

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    protected abstract HashSet<Matrix> getStopGradients();

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    protected abstract HashSet<Matrix> getConstantMatrices();

    /**
     * Resets layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void resetLayer() throws MatrixException {
        if (procedure != null) procedure.reset();
        resetLayerOutputs();
    }

    /**
     * Reinitializes neural network layer.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void reinitialize() throws NeuralNetworkException, MatrixException {
        if (procedure != null) procedure.reset();
        resetLayerOutputs();
    }

    /**
     * Prepares forward process step.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return previous outputs.
     */
    protected Sequence prepareForwardProcess() throws MatrixException {
        resetLayer();
        return getPreviousLayer().isConvolutionalLayer() && !isConvolutionalLayer() ? getPreviousLayerOutputs().flatten() : getPreviousLayerOutputs();
    }

    /**
     * Takes single forward processing step to process layer input(s).<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardProcess() throws MatrixException, DynamicParamException {
        procedure.calculateExpression(prepareForwardProcess(), getLayerOutputs());
    }

    /**
     * Takes single backward processing step to process layer output gradient(s) towards input.<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void backwardProcess() throws MatrixException, DynamicParamException {
        resetLayerGradients();
        executeBackwardProcess(isConvolutionalLayer() && hasNextLayer() && !getNextLayer().isConvolutionalLayer() ? getNextLayerGradients().unflatten(getLayerWidth(), getLayerHeight(), getLayerDepth()) : getNextLayerGradients());
        updateWeightGradients();
    }

    /**
     * Executes backward process step.
     *
     * @param nextLayerGradients next layer gradients.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void executeBackwardProcess(Sequence nextLayerGradients) throws MatrixException, DynamicParamException {
        if (procedure != null) procedure.calculateGradient(nextLayerGradients, getLayerGradients(), -1);
    }

    /**
     * Registers weights of layer.
     *
     * @param weight weight matrix to be registered.
     * @param forRegularization true if weight is registered for regularization otherwise false.
     * @param forNormalization true if weight is registered for normalization otherwise false.
     */
    public void registerWeight(Matrix weight, boolean forRegularization, boolean forNormalization) {
        weights.add(weight);
        weightsMap.put(weightsMap.size(), weight);
        if (forNormalization) normalizedWeights.add(weight);
        if (forRegularization) regularizedWeights.add(weight);
    }

    /**
     * Calculates gradient sum for wights after backward propagation step.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void updateWeightGradients() throws MatrixException {
        weightGradientSums = new HashMap<>();
        for (Matrix weight : weights) weightGradientSums.put(weight, procedure.getGradient(weight));
    }

    /**
     * Returns set of weights.
     *
     * @return set of weights.
     */
    public HashSet<Matrix> getWeights() {
        return weights;
    }

    /**
     * Returns map of weights.
     *
     * @return map of weights.
     */
    public HashMap<Integer, Matrix> getWeightsMap() {
        return weightsMap;
    }

    /**
     * Returns weights for normalization.
     *
     * @return weights for normalization.
     */
    public HashSet<Matrix> getNormalizedWeights() {
        return normalizedWeights;
    }

    /**
     * Returns weights for regularization.
     *
     * @return weights for regularization.
     */
    public HashSet<Matrix> getRegularizedWeights() {
        return regularizedWeights;
    }

    /**
     * Returns neural network weight gradients.
     *
     * @return neural network weight gradients.
     */
    public HashMap<Matrix, Matrix> getLayerWeightGradients() {
        return weightGradientSums;
    }

    /**
     * Returns width of previous layer.
     *
     * @return width of previous layer.
     */
    public int getPreviousLayerWidth() {
        return getPreviousLayer().isConvolutionalLayer() && !isConvolutionalLayer() ? getPreviousLayer().getLayerWidth() * getPreviousLayer().getLayerHeight() * getPreviousLayer().getLayerDepth() : getPreviousLayer().getLayerWidth();
    }

    /**
     * Returns height of previous layer.
     *
     * @return height of previous layer.
     */
    public int getPreviousLayerHeight() {
        return getPreviousLayer().isConvolutionalLayer() && !isConvolutionalLayer() ? 1 : getPreviousLayer().getLayerHeight();
    }

    /**
     * Returns depth of previous layer.
     *
     * @return depth of previous layer.
     */
    public int getPreviousLayerDepth() {
        return getPreviousLayer().isConvolutionalLayer() && !isConvolutionalLayer() ? 1 : getPreviousLayer().getLayerDepth();
    }

    /**
     * Returns width of next layer.
     *
     * @return width of next layer.
     */
    public int getNextLayerWidth() {
        return getNextLayer().getLayerWidth();
    }

    /**
     * Returns height of next layer.
     *
     * @return height of next layer.
     */
    public int getNextLayerHeight() {
        return getNextLayer().getLayerHeight();
    }

    /**
     * Returns depth of next layer.
     *
     * @return depth of next layer.
     */
    public int getNextLayerDepth() {
        return getNextLayer().getLayerDepth();
    }

    /**
     * Sets optimizer for layer.<br>
     * Optimizer optimizes weight parameters iteratively towards optimal solution.<br>
     *
     * @param optimizer optimizer to be added.
     */
    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    /**
     * Resets optimizer for layer.
     *
     */
    public void resetOptimizer() {
        optimizer.reset();
    }

    /**
     * Returns true if neural network is in training mode otherwise false.
     *
     * @return true if neural network is in training mode otherwise false.
     */
    public boolean isTraining() {
        return isTraining;
    }

    /**
     * Sets training flag.
     *
     * @param isTraining if true layer is training otherwise false.
     */
    protected void setTraining(boolean isTraining) {
        this.isTraining = isTraining;
    }

    /**
     * Resets optimizer of layer.
     *
     */
    public void reset() {
        resetOptimizer();
    }

    /**
     * Executes weight updates with regularizers and optimizer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void optimize() throws MatrixException, DynamicParamException {
        for (Matrix weight : weightGradientSums.keySet()) optimizer.optimize(weight, weightGradientSums.get(weight));
    }

    /**
     * Cumulates error from (L1 / L2 / Lp) regularization.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return cumulated error from regularization.
     */
    public double error() throws MatrixException, DynamicParamException {
        return 0;
    }

    /**
     * Returns number of layer parameters.
     *
     * @return number of layer parameters.
     */
    protected int getNumberOfParameters() {
        int numberOfParameters = 0;
        for (Matrix weight : weights) numberOfParameters += weight.size();
        return numberOfParameters;
    }

    /**
     * Appends other neural network layer with equal weights to this layer by weighting factor tau.
     *
     * @param otherNeuralNetworkLayer other neural network layer.
     * @param tau tau which controls contribution of other layer.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void append(NeuralNetworkLayer otherNeuralNetworkLayer, double tau) throws MatrixException {
        HashMap<Integer, Matrix> otherWeightsMap = otherNeuralNetworkLayer.getWeightsMap();
        for (Integer index : weightsMap.keySet()) {
            Matrix weight = weightsMap.get(index);
            Matrix otherWeight = otherWeightsMap.get(index);
            weight.multiply(1 - tau).add(otherWeight.multiply(tau), weight);
        }
    }

    /**
     * Returns optimizer by name.
     *
     * @return optimizer by name.
     */
    protected String getOptimizerByName() {
        return "Optimizer: " + optimizer.getName();
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected abstract String getLayerDetailsByName();

    /**
     * Prints structure and metadata of neural network layer.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void print() throws NeuralNetworkException {
        System.out.println(getLayerName() + " [ Width: " + getLayerWidth() + ", Height: " + getLayerHeight() + ", Depth: " + getLayerDepth() + " ]");
        System.out.println("Number of parameters: " + getNumberOfParameters());
        System.out.println(getOptimizerByName());
        String layerDetailsByName = getLayerDetailsByName();
        if (layerDetailsByName != null) System.out.println("Layer details [ " + layerDetailsByName + " ]");
    }

    /**
     * Prints forward expression chains of layer.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void printExpressions() throws NeuralNetworkException {
        System.out.println(getLayerName() + ": ");
        procedure.printExpressionChain();
        System.out.println();
    }

    /**
     * Prints backward gradient chains of layer.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void printGradients() throws NeuralNetworkException {
        System.out.println(getLayerName() + ": ");
        procedure.printGradientChain();
        System.out.println();
    }

}
