/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.layer;

import core.NeuralNetworkException;
import core.normalization.Normalization;
import core.normalization.NormalizationFactory;
import core.normalization.NormalizationType;
import core.optimization.OptimizationType;
import core.optimization.Optimizer;
import core.optimization.OptimizerFactory;
import core.regularization.Regularization;
import core.regularization.RegularizationFactory;
import core.regularization.RegularizationType;
import utils.*;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeMap;

/**
 * Connector class that connects two succeeding layers to each other.<br>
 * Connector always assumes that it is responsible for handling weights and bias of next layer.<br>
 * Handles weight and bias registration, gradient summing for next layer.<br>
 * Coordinates normalization, weight and bias regularization and optimization for next layer.<br>
 *
 */
public class Connector implements Serializable {

    private static final long serialVersionUID = -5803059922154462113L;

    /**
     * Reference to previous neural network layer.
     *
     */
    private AbstractLayer pLayer;

    /**
     * Reference to next neural network layer.
     *
     */
    private AbstractLayer nLayer;

    /**
     * Set of weights to be managed.
     *
     */
    private final HashSet<Matrix> Ws = new HashSet<>();

    /**
     * Map of gradients of weights to be managed.
     *
     */
    private final HashMap<Matrix, TreeMap<Integer, Matrix>> dWs = new HashMap<>();

    /**
     * Gradient sum of weights.
     *
     */
    private HashMap<Matrix, Matrix> dWSums;

    /**
     * Weights registered for optimization.
     *
     */
    private final HashSet<Matrix> opt = new HashSet<>();

    /**
     * Weights registered for regularization.
     *
     */
    private final HashSet<Matrix> reg = new HashSet<>();

    /**
     * Weights registered for normalization.
     *
     */
    private final HashSet<Matrix> norm = new HashSet<>();

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
     * Optimizer for the connector.
     *
     */
    private Optimizer optimizer = OptimizerFactory.create(OptimizationType.ADAM);

    /**
     * If true neural network is in training mode otherwise false.
     *
     */
    private transient boolean training;

    /**
     * Default constructor for connector.
     *
     * @param pLayer reference to previous layer.
     * @param nLayer reference to next layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Connector(AbstractLayer pLayer, AbstractLayer nLayer) throws DynamicParamException {
        this.pLayer = pLayer;
        this.nLayer = nLayer;
    }

    /**
     * Registers weights of next layer.
     *
     * @param W weight matrix to be registered.
     * @param forOptimization true if weight is registered for optimization otherwise false.
     * @param forRegularization true if weight is registered for regularization otherwise false.
     * @param forNormalization true if weight is registered for normalization otherwise false.
     */
    public void registerWeight(Matrix W, boolean forOptimization, boolean forRegularization, boolean forNormalization) {
        Ws.add(W);
        dWs.put(W, new TreeMap<>());
        if (forOptimization) opt.add(W);
        if (forRegularization) reg.add(W);
        if (forNormalization) norm.add(W);
    }

    /**
     * Calculates gradient sum after backward propagation step.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void sumGrad() throws MatrixException {
        dWSums = new HashMap<>();
        for (Matrix W : dWs.keySet()) {
            TreeMap<Integer, Matrix> dW = dWs.get(W);
            Matrix dWSum = new DMatrix(dW.get(dW.firstKey()).getRows(), dW.get(dW.firstKey()).getCols());
            for (Matrix dWItem : dW.values()) dWSum.add(dWItem, dWSum);
            dWSums.put(W, dWSum);
        }
    }

    /**
     * Resets gradients of weights.
     *
     */
    public void resetGrad() {
        for (Matrix W : dWs.keySet()) dWs.put(W, new TreeMap<>());
    }

    /**
     * Returns set of weights.
     *
     * @return set of weights.
     */
    public HashSet<Matrix> getWs() {
        return Ws;
    }

    /**
     * Returns map of gradients of weights.
     *
     * @return map of gradients of weights.
     */
    public HashMap<Matrix, TreeMap<Integer, Matrix>> getdWs() {
        return dWs;
    }

    /**
     * Returns gradients of weights for weight W.
     *
     * @param W weight
     * @return gradients of weights.
     */
    public TreeMap<Integer, Matrix> getdWs(Matrix W) {
        return dWs.get(W);
    }

    /**
     * Returns gradient sum for weight W.
     *
     * @param W weight
     * @return gradient sum.
     */
    public Matrix getdWsSums(Matrix W) {
        return dWSums.get(W);
    }

    /**
     * Returns set of weights registered for optimization.
     *
     * @return set of weights registered for optimization.
     */
    public HashSet<Matrix> getOpt() {
        return opt;
    }

    /**
     * Returns set of weights registered for regularization.
     *
     * @return set of weights registered for regularization.
     */
    public HashSet<Matrix> getReg() {
        return reg;
    }

    /**
     * Returns set of weights registered for normalization.
     *
     * @return set of weights registered for normalization.
     */
    public HashSet<Matrix> getNorm() {
        return norm;
    }

    /**
     * Adds regularization method for connector (next layer).
     *
     * @param regularizationType regularization method.
     * @throws NeuralNetworkException throws exception if adding of regularizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addRegularization(RegularizationType regularizationType) throws NeuralNetworkException, DynamicParamException {
        addRegularization(regularizationType, null);
    }

    /**
     * Adds regularization method for connector (next layer).
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
        Regularization regularizer = RegularizationFactory.create(regularizationType, this, nLayer != null, params);
        regularizers.add(regularizer);
    }

    /**
     * Removes any regularization from the connector (next layer).
     *
     */
    public void removeRegularization() {
        regularizers.clear();
    }

    /**
     * Removes specific regularization from the connector (next layer).
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
     * Returns set of regularization methods.
     *
     * @return set of regularization methods.
     */
    public HashSet<Regularization> getRegularization() {
        return regularizers;
    }

    /**
     * Resets specific regularization for the connector (next layer).
     *
     * @param regularizationType regularization method to be reset.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if reset of regularizer fails.
     */
    public void resetRegularization(RegularizationType regularizationType) throws MatrixException, NeuralNetworkException {
        Regularization resetRegularization = null;
        for (Regularization regularization : regularizers) {
            if (RegularizationFactory.getRegularizationType(regularization) == regularizationType) {
                resetRegularization = regularization;
            }
        }
        if (resetRegularization != null) resetRegularization.reset();
    }

    /**
     * Resets all regularization for the connector (next layer).
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void resetRegularization() throws MatrixException {
        for (Regularization regularization : regularizers) regularization.reset();
    }

    /**
     * Adds normalization method for connector (next layer).
     *
     * @param normalizationType normalization method.
     * @throws NeuralNetworkException throws exception if adding of normalizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addNormalization(NormalizationType normalizationType) throws NeuralNetworkException, DynamicParamException {
        addNormalization(normalizationType, null);
    }

    /**
     * Adds normalization method for connector (next layer).
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
        Normalization normalizer = NormalizationFactory.create(normalizationType, this, params);
        normalizers.add(normalizer);
    }

    /**
     * Removes any normalization from the connector (next layer).
     *
     */
    public void removeNormalization() {
        normalizers.clear();
    }

    /**
     * Removes specific normalization from the connector (next layer).
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
     * Returns set of normalization methods.
     *
     * @return set of normalization methods.
     */
    public HashSet<Normalization> getNormalization() {
        return normalizers;
    }

    /**
     * Resets specific normalization for the connector (next layer).
     *
     * @param normalizationType normalization method to be reset.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if reset of normalizer fails.
     */
    public void resetNormalization(NormalizationType normalizationType) throws MatrixException, NeuralNetworkException {
        Normalization resetNormalization = null;
        for (Normalization normalization : normalizers) {
            if (NormalizationFactory.getNormalizationType(normalization) == normalizationType) {
                resetNormalization = normalization;
            }
        }
        if (resetNormalization != null) resetNormalization.reset();
    }

    /**
     * Resets all normalization for the connector (next layer).
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void resetNormalization() throws MatrixException {
        for (Normalization normalization : normalizers) normalization.reset();
    }

    /**
     * Sets optimizer for connector (next layer).<br>
     * Optimizer optimizes weight parameters iteratively towards optimal solution.<br>
     *
     * @param optimizer optimizer to be added.
     */
    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    /**
     * Gets optimizer for connector (next layer).
     *
     * @return optimizer for connector (next layer).
     */
    public Optimizer getOptimizer() {
        return optimizer;
    }

    /**
     * Resets optimizer for connector (next layer).
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void resetOptimizer() throws MatrixException {
        if (optimizer!= null) optimizer.reset();
    }

    /**
     * Sets relative size of mini batch.
     *
     * @param miniBatchFactor relative size of mini batch.
     */
    public void setMiniBatchFactor(double miniBatchFactor) {
        if (optimizer!= null) optimizer.setMiniBatchFactor(miniBatchFactor);
        getNLayer().setMiniBatchFactor(miniBatchFactor);
    }

    /**
     * Returns true if neural network is in training mode otherwise false.
     *
     * @return true if neural network is in training mode otherwise false.
     */
    public boolean isTraining() {
        return training;
    }

    /**
     * Resets regularizers, normalizers and optimizer of the connector.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void reset() throws MatrixException {
        for (Regularization regularization : regularizers) regularization.reset();
        for (Normalization normalization : normalizers) normalization.reset();
        optimizer.reset();
    }

    /**
     * Gets reference to previous layer.
     *
     * @return reference to previous layer.
     */
    public AbstractLayer getPLayer() {
        return pLayer;
    }

    /**
     * Returns true if connector has previous layer otherwise false.
     *
     * @return true if connector has previous layer otherwise false.
     */
    public boolean hasPLayer() {
        return (pLayer != null);
    }

    /**
     * Gets reference to next layer.
     *
     * @return reference to next layer.
     */
    public AbstractLayer getNLayer() {
        return nLayer;
    }

    /**
     * Returns true if connector has next layer otherwise false.
     *
     * @return true if connector has next layer otherwise false.
     */
    public boolean hasNLayer() {
        return (nLayer != null);
    }

    /**
     * Starts next layer.
     *
     * @throws NeuralNetworkException throws exception if starting of next layer fails.
     */
    public void start() throws NeuralNetworkException {
        getNLayer().start();
    }

    /**
     * Stops next layer.
     *
     */
    public void stop() {
        getNLayer().stop();
    }

    /**
     * Sends state completed signal for previous or next layer depending on state of forwardDirection parameter.
     *
     * @param forwardDirection if true sends the signal to next layer otherwise send the signal previous layer.
     */
    public void stateCompleted(boolean forwardDirection) {
        if (forwardDirection) getNLayer().stateCompleted(true);
        else getPLayer().stateCompleted(false);
    }

    /**
     * Executes training step for next layer.
     *
     */
    public void train() {
        training = true;
        for (Regularization regularizer : regularizers) regularizer.setTraining(training);
        for (Normalization normalizer : normalizers) normalizer.setTraining(training);
        getNLayer().train();
    }

    /**
     * Executes predict step for next layer.
     *
     */
    public void predict() {
        training = false;
        for (Regularization regularizer : regularizers) regularizer.setTraining(training);
        for (Normalization normalizer : normalizers) normalizer.setTraining(training);
        getNLayer().predict();
    }

    /**
     * Gets output of next layer.
     *
     * @return output of next layer.
     */
    public TreeMap<Integer, Matrix> getOutput() {
        return getNLayer().getOutput();
    }

    /**
     * Executes regularization methods with forward step at the step start.<br>
     * This operation assumes regularization prior forward step.<br>
     *
     * @param ins input samples for forward step.
     * @param index if index is zero or positive value operation is executed for this sample. if index is -1 operation is executed for all samples.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void regulateForwardPre(TreeMap<Integer, Matrix> ins, int index) throws MatrixException {
        for (Regularization regularizer : regularizers) regularizer.forwardPre(ins, index);
    }

    /**
     * Executes normalization methods with forward step at the step start.<br>
     * This operation assumes normalization prior forward step.<br>
     *
     * @param ins input samples for forward step.
     * @param depthIn number of channels of a convolutional layer.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void normalizeForwardPre(TreeMap<Integer, Matrix> ins, int depthIn) throws MatrixException {
        for (Normalization normalizer : normalizers) normalizer.forwardPre(ins, depthIn);
    }

    /**
     * Executes regularization methods with forward step at the step end.<br>
     * This operation assumes regularization post forward step.<br>
     *
     * @param outs output samples for forward step.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void regulateForwardPost(TreeMap<Integer, Matrix> outs) throws MatrixException {
        if (getNLayer() instanceof OutputLayer) return;
        for (Regularization regularizer : regularizers) regularizer.forwardPost(outs);
    }

    /**
     * Executes normalization methods with forward step at the step end.<br>
     * This operation assumes normalization post forward step.<br>
     *
     * @param outs output samples for forward step.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void normalizeForwardPost(TreeMap<Integer, Matrix> outs) throws MatrixException {
        if (getNLayer() instanceof OutputLayer) return;
        for (Normalization normalizer : normalizers) normalizer.forwardPost(outs);
    }

    /**
     * Executes backward phase for training step.
     *
     * @throws NeuralNetworkException throws exception if backward operation fails.
     */
    public void backward() throws NeuralNetworkException {
        getPLayer().backward();
    }

    /**
     * Executes regularization methods with backward phase of training step at pre in.
     *
     * @param index if index is zero or positive value operation is executed for this sample. if index is -1 operation is executed for all samples.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void regulateBackward(int index) throws MatrixException {
        for (Regularization regularizer : regularizers) regularizer.backward(index);
    }

    /**
     * Executes regularization and normalization methods with backward phase of training step at post in.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void normalizeBackward() throws MatrixException {
        for (Normalization normalizer : normalizers) normalizer.backward();
    }

    /**
     * Executes weight updates with regularizers and optimizer.
     *
     */
    public void update() {
        try {
            for (Regularization regularizer : regularizers) regularizer.update();
            for (Matrix W : opt) optimizer.optimize(W, dWSums.get(W));
        }
        catch (MatrixException exception) {
            System.out.println(exception.toString());
            System.exit(-1);
        }
        getNLayer().update();
    }

    /**
     * Cumulates error from regularization. Mainly from L1 / L2 / Lp regularization.
     *
     * @return cumulated error from regularization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double error() throws MatrixException {
        double error = 0;
        for (Regularization regularizer : regularizers) error += regularizer.error();
        if (pLayer != null) {
            if (pLayer.getBackward() != null) error += pLayer.getBackward().error();
        }
        return error;
    }

}
