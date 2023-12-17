/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer;

import core.network.NeuralNetworkException;
import core.optimization.Optimizer;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
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
     * Constant matrices for layer.
     *
     */
    private HashSet<Matrix> constantMatrices;

    /**
     * Stop gradient matrices for layer.
     *
     */
    private HashSet<Matrix> stopGradients;

    /**
     * If true neural network is in training mode otherwise false.
     *
     */
    private transient boolean isTraining;

    /**
     * If true procedure expression dependencies are reset otherwise false.
     *
     */
    private boolean resetDependencies = true;

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
        if (procedure != null) procedure.setActive(isTraining);
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
     * @throws MatrixException throws exception if layer dimensions are not matching.
     */
    protected abstract void initializeWeights() throws MatrixException;

    /**
     * Checks if layer is recurrent layer type.
     *
     * @return always false.
     */
    public boolean isRecurrentLayer() { return false; }

    /**
     * Checks if layer works with recurrent layers.
     *
     * @return if true layer works with recurrent layers otherwise false.
     */
    public boolean worksWithRecurrentLayer() {
        return true;
    }

    /**
     * Check if layer input is reversed.
     *
     * @return if true input layer input is reversed otherwise not.
     */
    public boolean isReversedInput() { return false; }

    /**
     * Returns true if input is joined otherwise returns false.
     *
     * @return true if input is joined otherwise returns false.
     */
    public boolean isJoinedInput() {
        return false;
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException       throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void defineProcedure() throws MatrixException, DynamicParamException, NeuralNetworkException {
        if (procedure == null) initializeWeights();
        procedure = new ProcedureFactory().getProcedure(this);
    }

    /**
     * Returns parameter matrices.
     *
     * @return parameter matrices.
     */
    public HashSet<Matrix> getParameterMatrices() {
        return getWeightSet() != null ? getWeightSet().getWeights() : null;
    }

    /**
     * Registers constant matrix.
     *
     * @param constantMatrix constant matrix.
     */
    protected void registerConstantMatrix(Matrix constantMatrix) {
        if (constantMatrices == null) constantMatrices = new HashSet<>();
        constantMatrices.add(constantMatrix);
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    public HashSet<Matrix> getConstantMatrices() {
        return constantMatrices;
    }

    /**
     * Registers stop gradient.
     *
     * @param stopGradient stop gradient.
     */
    protected void registerStopGradient(Matrix stopGradient) {
        if (stopGradients == null) stopGradients = new HashSet<>();
        stopGradients.add(stopGradient);
    }

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    public HashSet<Matrix> getStopGradients() {
        return stopGradients;
    }

    /**
     * Sets reset flag for procedure expression dependencies.
     *
     * @param resetDependencies if true procedure expression dependencies are reset otherwise false.
     */
    public void resetDependencies(boolean resetDependencies) {
        this.resetDependencies = resetDependencies;
    }

    /**
     * Resets layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void reset() throws MatrixException {
        super.reset();
        if (procedure != null) {
            procedure.reset();
            procedure.resetDependencies(isTraining() || resetDependencies);
        }
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
        if (procedure != null) procedure.calculateExpression(getInputSequences(), getLayerOutputs());
    }

    /**
     * Takes single backward processing step to process layer output gradient(s) towards input.<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void backwardProcess() throws MatrixException, DynamicParamException {
        if (procedure != null) procedure.calculateGradient(getLayerOutputGradients(), getInputGradientSequences(), getTruncateSteps());
    }

    /**
     * Returns number of truncated steps for gradient calculation. -1 means no truncation.
     *
     * @return number of truncated steps.
     */
    protected int getTruncateSteps() {
        return -1;
    }

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
        if (procedure != null) procedure.setOptimizer(optimizer);
    }

    /**
     * Resets optimizer of layer.
     *
     */
    public void resetOptimizer() {
        if (procedure != null) procedure.resetOptimizer();
    }

    /**
     * Executes weight updates with regularizers and optimizer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void optimize() throws MatrixException, DynamicParamException {
        if (procedure != null) procedure.optimize();
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
            entry.getValue().multiply(1 - tau).addBy(otherNeuralNetworkWeightsMap.get(entry.getKey()).multiply(tau));
        }
    }

    /**
     * Returns number of layer parameters.
     *
     * @return number of layer parameters.
     */
    public int getNumberOfParameters() {
        return getWeightSet() != null ? getWeightSet().getNumberOfParameters() : 0;
    }

    /**
     * Returns optimizer by name.
     *
     * @return optimizer by name.
     */
    protected String getOptimizerByName() {
        return "Optimizer: " + (procedure != null ? procedure.getOptimizerByName() : "N/A");
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
        String layerConnections = hasPreviousLayers() ? getLayerConnections() : "";
        String layerDetailsByName = getLayerDetailsByName();
        if (layerDetailsByName != null) System.out.println("Layer details [ " + layerConnections + (!layerDetailsByName.equals("") ? ", " + layerDetailsByName : "") + " ]");
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
