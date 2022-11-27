/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer;

import core.network.NeuralNetworkException;
import core.optimization.Optimizer;
import core.optimization.OptimizerFactory;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.procedure.ForwardProcedure;
import utils.procedure.Procedure;
import utils.procedure.ProcedureFactory;

import java.util.*;

/**
 * Implements abstract execution layer supporting actual neural network layers (feed forward, recurrent, convolutional layers etc.)<br>
 * Provides supportive functions for actual neural network layers.<br>
 * Supports automatic gradient i.e. backward gradient calculation for layers needing it.<br>
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
     * Optimizer for layer.
     *
     */
    protected Optimizer optimizer = OptimizerFactory.createDefault();

    /**
     * If true neural network is in training mode otherwise false.
     *
     */
    private transient boolean isTraining;

    /**
     * Constructor for abstract execution layer.
     *
     * @param layerIndex layer index
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
     * Returns weight set.
     *
     * @return weight set.
     */
    protected abstract WeightSet getWeightSet();

    /**
     * Initializes neural network layer weights.
     *
     */
    protected abstract void initializeWeights();

    /**
     * Handles birectional input.
     *
     * @param input input
     * @return handled input
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix handleBidirectionalInput(Matrix input) throws MatrixException {
        return handleBidirectionalInput(input, getLayerIndex() - 1);
    }

    /**
     * Handles birectional input.
     *
     * @param input input
     * @param previousLayerIndex previous layer index
     * @return handled input
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix handleBidirectionalInput(Matrix input, int previousLayerIndex) throws MatrixException {
        return getPreviousLayer(previousLayerIndex).isBidirectional() ? input.split(getPreviousLayerWidth(previousLayerIndex) / 2, true) : input;
    }

    /**
     * Handles birectional input.
     *
     * @param input input
     * @return handled input
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected MMatrix handleBidirectionalInput(MMatrix input) throws MatrixException {
        return getPreviousLayer().isBidirectional() ? input.split(getPreviousLayerWidth() / 2, true) : input;
    }

    /**
     * Returns true if input is joined otherwise returns false.
     *
     * @return true if input is joined otherwise returns false.
     */
    protected boolean isJoinedInput() {
        return false;
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void defineProcedure() throws MatrixException, DynamicParamException, NeuralNetworkException {
        addInputSequence(getLayerIndex() - 1);
        addOtherInputLayers();

        if (procedure == null) initializeWeights();
        Procedure reverseProcedure = getReverseProcedure();
        procedure = new ProcedureFactory().getProcedure(this, getWeightSet() != null ? getWeightSet().getWeights() : null, getConstantMatrices(), getStopGradients(), reverseProcedure, isJoinedInput());
    }

    /**
     * Returns reversed procedure.
     *
     * @return reversed procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected abstract Procedure getReverseProcedure() throws MatrixException, DynamicParamException;

    /**
     * Adds other input layers.
     *
     */
    protected void addOtherInputLayers() {
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
    public void reset() throws MatrixException {
        super.reset();
        if (procedure != null) procedure.reset();
    }

    /**
     * Reinitializes neural network layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void reinitialize() throws MatrixException {
        reset();
        if (getWeightSet() != null) getWeightSet().reinitialize();
    }

    /**
     * Takes single forward processing step to process layer input(s).<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardProcess() throws MatrixException, DynamicParamException {
        reset();
        if (procedure != null) procedure.calculateForwardExpression(getInputSequences(), getLayerOutputs());
    }

    /**
     * Takes single backward processing step to process layer output gradient(s) towards input.<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void backwardProcess() throws MatrixException, DynamicParamException {
        if (procedure != null) procedure.calculateBackwardGradient(getNextLayerGradients(), getInputGradientSequences(), getTruncateSteps());
    }

    /**
     * Returns number of truncated steps for gradient calculation. -1 means no truncation.
     *
     * @return number of truncated steps.
     */
    protected abstract int getTruncateSteps();

    /**
     * Registers weights of layer.
     *
     * @param weight weight matrix to be registered.
     * @param forRegularization true if weight is registered for regularization otherwise false.
     * @param forNormalization true if weight is registered for normalization otherwise false.
     */
    public void registerWeight(Matrix weight, boolean forRegularization, boolean forNormalization) {
        weightsMap.put(weightsMap.size(), weight);
        if (forNormalization) normalizedWeights.add(weight);
        if (forRegularization) regularizedWeights.add(weight);
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
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public HashMap<Matrix, Matrix> getLayerWeightGradients() throws MatrixException {
        return procedure != null ? procedure.getGradients() : new HashMap<>();
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
     * Resets optimizer of layer.
     *
     */
    public void resetOptimizer() {
        optimizer.reset();
    }

    /**
     * Executes weight updates with regularizers and optimizer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void optimize() throws MatrixException, DynamicParamException {
        for (Map.Entry<Matrix, Matrix> entry : getLayerWeightGradients().entrySet()) optimizer.optimize(entry.getKey(), entry.getValue());
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
     * Appends other neural network layer with equal weights to this layer by weighting factor tau.
     *
     * @param otherNeuralNetworkLayer other neural network layer.
     * @param tau tau which controls contribution of other layer.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void append(NeuralNetworkLayer otherNeuralNetworkLayer, double tau) throws MatrixException {
        HashMap<Integer, Matrix> otherNeuralNetworkWeightsMap = otherNeuralNetworkLayer.getWeightsMap();
        for (Map.Entry<Integer, Matrix> entry : weightsMap.entrySet()) {
            entry.getValue().multiply(1 - tau).add(otherNeuralNetworkWeightsMap.get(entry.getKey()).multiply(tau), entry.getValue());
        }
    }

    /**
     * Returns number of layer parameters.
     *
     * @return number of layer parameters.
     */
    protected int getNumberOfParameters() {
        return getWeightSet() != null ? getWeightSet().getNumberOfParameters() : 0;
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
        if (procedure != null) procedure.printExpressionChain();
        else {
            System.out.print("N/A");
            System.out.println();
        }
        System.out.println();
    }

    /**
     * Prints backward gradient chains of layer.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void printGradients() throws NeuralNetworkException {
        System.out.println(getLayerName() + ": ");
        if (procedure != null) procedure.printGradientChain();
        else {
            System.out.print("N/A");
            System.out.println();
        }
        System.out.println();
    }

}
