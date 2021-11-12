/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer;

import core.network.NeuralNetworkException;
import core.normalization.Normalization;
import core.normalization.NormalizationFactory;
import core.normalization.NormalizationType;
import core.optimization.OptimizationType;
import core.optimization.Optimizer;
import core.optimization.OptimizerFactory;
import core.regularization.Regularization;
import core.regularization.RegularizationFactory;
import core.regularization.RegularizationType;
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
     * List of regularizers.
     *
     */
    private final HashSet<Regularization> regularizers = new HashSet<>();

    /**
     * List of normalizers.
     *
     */
    private final HashSet<Normalization> normalizers = new HashSet<>();

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
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void defineProcedure() throws MatrixException, DynamicParamException {
        procedure = new ProcedureFactory().getProcedure(this, getAllConstantMatrices());
        procedure.setNormalizers(getNormalization());
        procedure.setRegularizers(getRegularization());
        procedure.initialize();
        procedure.setStopGradient(getStopGradients(), true);
    }

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
     * Resets layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void resetLayer() throws MatrixException {
        procedure.reset();
        resetLayerOutputs();
        resetNormalization();
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
        reinitializeNormalization();
    }

    /**
     * Takes single forward processing step to process layer input(s).<br>
     * Additionally applies any normalization or regularization defined for layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardProcess() throws MatrixException, DynamicParamException {
        executeForwardProcess(prepareForwardProcess());
    }

    /**
     * Executes forward process step.
     *
     * @param previousOutputs outputs of previous layer.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void executeForwardProcess(Sequence previousOutputs) throws MatrixException, DynamicParamException {
        procedure.calculateExpression(previousOutputs, getLayerOutputs());
    }

    /**
     * Takes single backward processing step to process layer output gradient(s) towards input.<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     * Additionally applies any regularization defined for layer.<br>
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
        procedure.calculateGradient(nextLayerGradients, getLayerGradients(), -1);
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
        weight.setRegularize(forRegularization);
        weight.setNormalize(forNormalization);
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
     * Adds regularization method for layer.
     *
     * @param regularizationType regularization method.
     * @throws NeuralNetworkException throws exception if adding of regularizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addRegularization(RegularizationType regularizationType) throws NeuralNetworkException, DynamicParamException {
        addRegularization(regularizationType, null);
    }

    /**
     * Adds regularization method for layer.
     *
     * @param regularizationType regularization method.
     * @param params parameters for regularizer.
     * @throws NeuralNetworkException throws exception if adding of regularizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addRegularization(RegularizationType regularizationType, String params) throws NeuralNetworkException, DynamicParamException {
        for (Regularization regularization : regularizers) {
            if (RegularizationFactory.getRegularizationType(regularization) == regularizationType) throw new NeuralNetworkException("Regularizer: " + regularizationType + " already exists");
        }
        Regularization regularizer = RegularizationFactory.create(regularizationType, params);
        regularizers.add(regularizer);
    }

    /**
     * Removes any regularization from layer.
     *
     */
    public void removeRegularization() {
        regularizers.clear();
    }

    /**
     * Removes specific regularization from layer.
     *
     * @param regularizationType regularization method to be removed.
     * @throws NeuralNetworkException throws exception if removal of regularizer fails.
     */
    public void removeRegularization(RegularizationType regularizationType) throws NeuralNetworkException {
        Regularization removeRegularization = null;
        for (Regularization regularization : regularizers) {
            if (RegularizationFactory.getRegularizationType(regularization) == regularizationType) {
                removeRegularization = regularization;
            }
        }
        if (removeRegularization != null) regularizers.remove(removeRegularization);
    }

    /**
     * Returns set of regularization methods applied to layer.
     *
     * @return set of regularization methods applied to layer.
     */
    public HashSet<Regularization> getRegularization() {
        return regularizers;
    }

    /**
     * Adds normalization method for layer.
     *
     * @param normalizationType normalization method.
     * @throws NeuralNetworkException throws exception if adding of normalizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addNormalization(NormalizationType normalizationType) throws NeuralNetworkException, DynamicParamException {
        addNormalization(normalizationType, null);
    }

    /**
     * Adds normalization method for layer.
     *
     * @param normalizationType normalization method.
     * @param params parameters for normalizer.
     * @throws NeuralNetworkException throws exception if adding of normalizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addNormalization(NormalizationType normalizationType, String params) throws NeuralNetworkException, DynamicParamException {
        for (Normalization normalization : normalizers) {
            if (NormalizationFactory.getNormalizationType(normalization) == normalizationType) throw new NeuralNetworkException("Normalizer: " + normalizationType + " already exists");
        }
        Normalization normalizer = NormalizationFactory.create(normalizationType, params);
        normalizers.add(normalizer);
        if (optimizer != null) normalizer.setOptimizer(optimizer);
    }

    /**
     * Removes any normalization from layer.
     *
     */
    public void removeNormalization() {
        normalizers.clear();
    }

    /**
     * Removes specific normalization from layer.
     *
     * @param normalizationType normalization method to be removed.
     * @throws NeuralNetworkException throws exception if removal of normalizer fails.
     */
    public void removeNormalization(NormalizationType normalizationType) throws NeuralNetworkException {
        Normalization removeNormalization = null;
        for (Normalization normalization : normalizers) {
            if (NormalizationFactory.getNormalizationType(normalization) == normalizationType) {
                removeNormalization = normalization;
            }
        }
        if (removeNormalization != null) normalizers.remove(removeNormalization);
    }

    /**
     * Returns set of normalization methods applied to layer.
     *
     * @return set of normalization methods applied to layer.
     */
    public HashSet<Normalization> getNormalization() {
        return normalizers;
    }

    /**
     * Resets specific normalization for layer.
     *
     * @param normalizationType normalization method to be reset.
     * @throws NeuralNetworkException throws exception if reset of normalizer fails.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void resetNormalization(NormalizationType normalizationType) throws NeuralNetworkException, MatrixException {
        Normalization resetNormalization = null;
        for (Normalization normalization : normalizers) {
            if (NormalizationFactory.getNormalizationType(normalization) == normalizationType) {
                resetNormalization = normalization;
            }
        }
        if (resetNormalization != null) resetNormalization.reset();
    }

    /**
     * Reinitializes specific normalization for layer.
     *
     * @param normalizationType normalization method to be reinitialized.
     * @throws NeuralNetworkException throws exception if reinitialization of normalizer fails.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void reinitializeNormalization(NormalizationType normalizationType) throws NeuralNetworkException, MatrixException {
        Normalization reinitializeNormalization = null;
        for (Normalization normalization : normalizers) {
            if (NormalizationFactory.getNormalizationType(normalization) == normalizationType) {
                reinitializeNormalization = normalization;
            }
        }
        if (reinitializeNormalization != null) reinitializeNormalization.reinitialize();
    }

    /**
     * Resets all normalization for layer.
     *
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void resetNormalization() throws MatrixException {
        for (Normalization normalizer : normalizers) normalizer.reset();
    }

    /**
     * Reinitializes all normalization for layer.
     *
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void reinitializeNormalization() throws MatrixException {
        for (Normalization normalizer : normalizers) normalizer.reinitialize();
    }

    /**
     * Sets optimizer for layer.<br>
     * Optimizer optimizes weight parameters iteratively towards optimal solution.<br>
     *
     * @param optimizer optimizer to be added.
     */
    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
        for (Normalization normalizer: normalizers) normalizer.setOptimizer(optimizer);
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
        for (Regularization regularizer : regularizers) regularizer.setTraining(isTraining);
        for (Normalization normalizer : normalizers) normalizer.setTraining(isTraining);
    }

    /**
     * Resets normalizers and optimizer of layer.
     *
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void reset() throws MatrixException {
        resetNormalization();
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
        for (Normalization normalizer : normalizers) normalizer.optimize();
    }

    /**
     * Cumulates error from (L1 / L2 / Lp) regularization.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return cumulated error from regularization.
     */
    public double error() throws MatrixException, DynamicParamException {
        double error = procedure.getRegularizationError();
        if (getPreviousLayer() != null) error += getPreviousLayer().error();
        return error;
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
     * Returns regularizers by name.
     *
     * @return regularizers by name.
     */
    protected String getRegularizersByName() {
        StringBuilder regularizerNames = new StringBuilder("Regularizers: ");
        int index = 1;
        for (Regularization regularization : regularizers) {
            regularizerNames.append(regularization.getName());
            if (index < regularizers.size()) regularizerNames.append(", ");
        }
        return regularizerNames.toString();
    }

    /**
     * Returns normalizers by name.
     *
     * @return normalizers by name.
     */
    protected String getNormalizersByName() {
        StringBuilder normalizerNames = new StringBuilder("Normalizers: ");
        int index = 1;
        for (Normalization normalization : normalizers) {
            normalizerNames.append(normalization.getName());
            if (index < normalizers.size()) normalizerNames.append(", ");
        }
        return normalizerNames.toString();
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
        System.out.println(getNormalizersByName());
        System.out.println(getRegularizersByName());
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
        for (Normalization normalization : normalizers) normalization.printExpressions();
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
        for (Normalization normalization : normalizers) normalization.printGradients();
    }

}
