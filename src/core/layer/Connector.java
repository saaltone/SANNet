/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
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
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeMap;

/**
 * Connector class that connects two succeeding layers to each other.<br>
 * Connector always assumes that it is responsible for handling weights and bias of next layer.<br>
 * Handles weight and bias registration, gradient summing for next layer.<br>
 * Coordinates normalization, regularization and optimization for next layer.<br>
 *
 */
public class Connector implements Serializable {

    private static final long serialVersionUID = -5803059922154462113L;

    /**
     * Reference to previous neural network layer.
     *
     */
    private final AbstractLayer pLayer;

    /**
     * Reference to next neural network layer.
     *
     */
    private final AbstractLayer nLayer;

    /**
     * Set of weights to be managed.
     *
     */
    private final HashSet<Matrix> Ws = new HashSet<>();

    /**
     * Ordered map of weights.
     *
     */
    private final HashMap<Integer, Matrix> WsMap = new HashMap<>();

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
        WsMap.put(WsMap.size(), W);
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
            if (dW.size() > 0) {
                Matrix dWSum = new DMatrix(dW.get(dW.firstKey()).getRows(), dW.get(dW.firstKey()).getCols());
                for (Matrix dWItem : dW.values()) dWSum.add(dWItem, dWSum);
                dWSum.divide(dW.size(), dWSum);
                dWSums.put(W, dWSum);
            }
        }
    }

    /**
     * Resets gradients of weights.
     *
     */
    public void resetGrad() {
        dWs.replaceAll((w, v) -> new TreeMap<>());
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
     * Returns ordered map of weights.
     *
     * @return ordered map of weights.
     */
    public HashMap<Integer, Matrix> getWsMap() {
        return WsMap;
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
     * Returns width of previous layer.
     *
     * @return width of previous layer.
     */
    public int getPLayerWidth() {
        return pLayer.isConvolutionalLayer() && !nLayer.isConvolutionalLayer() ? pLayer.getWidth() * pLayer.getHeight() * pLayer.getDepth() : pLayer.getWidth();
    }

    /**
     * Returns height of previous layer.
     *
     * @return height of previous layer.
     */
    public int getPLayerHeight() {
        return pLayer.isConvolutionalLayer() && !nLayer.isConvolutionalLayer() ? 1 : pLayer.getHeight();
    }

    /**
     * Returns depth of previous layer.
     *
     * @return depth of previous layer.
     */
    public int getPLayerDepth() {
        return pLayer.isConvolutionalLayer() && !nLayer.isConvolutionalLayer() ? 1 : pLayer.getDepth();
    }

    /**
     * Returns width of next layer.
     *
     * @return width of next layer.
     */
    public int getNLayerWidth() {
        return nLayer.getWidth();
    }

    /**
     * Returns height of next layer.
     *
     * @return height of next layer.
     */
    public int getNLayerHeight() {
        return nLayer.getHeight();
    }

    /**
     * Returns depth of next layer.
     *
     * @return depth of next layer.
     */
    public int getNLayerDepth() {
        return nLayer.getDepth();
    }

    /**
     * Appends other connector with equal weights to this connector by weighted factor tau.
     *
     * @param otherConnector other connector
     * @param tau tau which controls contribution of other connector.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void append(Connector otherConnector, double tau) throws MatrixException {
        HashMap<Integer, Matrix> otherWsMap = otherConnector.getWsMap();
        for (Integer index : WsMap.keySet()) {
            Matrix W = WsMap.get(index);
            Matrix otherW = otherWsMap.get(index);
            W.multiply(1 - tau).add(otherW.multiply(tau), W);
        }
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
        Regularization regularizer = RegularizationFactory.create(regularizationType, params);
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
     * @throws NeuralNetworkException throws exception if reset of regularizer fails.
     */
    public void resetRegularization(RegularizationType regularizationType) throws NeuralNetworkException {
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
     */
    public void resetRegularization() {
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
        Normalization normalizer = NormalizationFactory.create(normalizationType, params);
        normalizers.add(normalizer);
        normalizer.setNormalizableParameters(norm);
        if (optimizer != null) normalizer.setOptimizer(optimizer);
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
     * @throws NeuralNetworkException throws exception if reset of normalizer fails.
     */
    public void resetNormalization(NormalizationType normalizationType) throws NeuralNetworkException {
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
     */
    public void resetNormalization() {
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
        for (Normalization normalizer: normalizers) normalizer.setOptimizer(optimizer);
    }

    /**
     * Returns optimizer for connector (next layer).
     *
     * @return optimizer for connector (next layer).
     */
    public Optimizer getOptimizer() {
        return optimizer;
    }

    /**
     * Resets optimizer for connector (next layer).
     *
     */
    public void resetOptimizer() {
        if (optimizer!= null) optimizer.reset();
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
     */
    public void reset() {
        for (Regularization regularization : regularizers) regularization.reset();
        for (Normalization normalization : normalizers) normalization.reset();
        optimizer.reset();
    }

    /**
     * Returns reference to previous layer.
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
     * Returns reference to next layer.
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
     * Sets if recurrent inputs of layer are allowed to be reset.
     *
     * @param allowLayerReset if true allows reset.
     */
    public void setAllowLayerReset(boolean allowLayerReset) {
        getNLayer().setAllowLayerReset(allowLayerReset);
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
     * Returns output of next layer.
     *
     * @return output of next layer.
     */
    public Sequence getOutput() {
        return getNLayer().getOutput();
    }

    /**
     * Executes regularization methods with forward step.<br>
     *
     * @param inputs input sequence
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void regulateForward(Sequence inputs) throws MatrixException {
        for (Regularization regularizer : regularizers) {
            regularizer.setMiniBatchSize(inputs.sampleSize());
            regularizer.forward(inputs);
        }
    }

    /**
     * Executes regularization methods with forward step.<br>
     *
     */
    public void regulateForward() {
        for (Regularization regularizer : regularizers) {
            for (Matrix W : reg) {
                regularizer.forward(W);
            }
        }
    }

    /**
     * Executes regularization methods with forward step.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void normalizeForward() throws MatrixException {
        for (Normalization normalizer : normalizers) {
            for (Matrix W : norm) {
                normalizer.forward(W);
            }
        }
    }

    /**
     * Executes regularization methods with forward step to finalize.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void normalizeFinalizeForward() throws MatrixException {
        for (Normalization normalizer : normalizers) {
            for (Matrix W : norm) {
                normalizer.forwardFinalize(W);
            }
        }
    }

    /**
     * Executes regularization methods with forward step to finalize.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void normalizeBackward() throws MatrixException {
        for (Normalization normalizer : normalizers) {
            for (Matrix W : norm) {
                normalizer.backward(W, getdWsSums(W));
            }
        }
    }

    /**
     * Resets normalizer state.
     *
     */
    public void normalizerReset() {
        for (Normalization normalizer : normalizers) normalizer.reset();
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
     * Executes regularization methods for backward phase of training step.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void regulateBackward() throws MatrixException {
        for (Regularization regularizer : regularizers) {
            for (Matrix W : reg) regularizer.backward(W, dWSums.get(W));
        }
    }

    /**
     * Executes weight updates with regularizers and optimizer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void update() throws MatrixException {
        for (Matrix W : opt) {
            if (dWSums.containsKey(W)) optimizer.optimize(W, dWSums.get(W));
        }
        getNLayer().update();
    }

    /**
     * Cumulates error from regularization. Mainly from L1 / L2 / Lp regularization.
     *
     * @return cumulated error from regularization.
     */
    public double error() {
        double error = 0;
        for (Regularization regularizer : regularizers) {
            for (Matrix W : reg) error += regularizer.error(W);
        }
        if (pLayer != null) {
            if (pLayer.getBackward() != null) error += pLayer.getBackward().error();
        }
        return error;
    }

}
