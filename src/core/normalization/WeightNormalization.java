/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.normalization;

import core.optimization.Optimizer;
import utils.*;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.procedure.Node;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Class that implements Weight Normalization for neural network layer.<br>
 * <br>
 * Reference: https://arxiv.org/pdf/1602.07868.pdf<br>
 *
 */
public class WeightNormalization implements Normalization, Serializable {

    private static final long serialVersionUID = 1741544680542755148L;

    /**
     * If true neural network is in state otherwise false.
     *
     */
    private transient boolean isTraining;

    /**
     * Normalizable parameters.
     *
     */
    private HashSet<Matrix> norm;

    /**
     * Tree map for un-normalized weights.
     *
     */
    private final HashMap<Matrix, Matrix> Ws = new HashMap<>();

    /**
     * Tree map for storing Weight normalizing factors (1 / sqrt(2-norm W))
     *
     */
    private final HashMap<Matrix, Double> iNorms = new HashMap<>();

    /**
     * Tree map for storing Identity matrices for backward calculation.
     *
     */
    private final HashMap<Matrix, Matrix> Is = new HashMap<>();

    /**
     * Weight normalization scalar.
     *
     */
    private double g = 1;

    /**
     * Constructor for Weight normalization class.
     *
     */
    public WeightNormalization() {
    }

    /**
     * Constructor for Weight normalization class.
     *
     * @param params parameters for Weight normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public WeightNormalization(String params) throws DynamicParamException {
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for Weight normalization.
     *
     * @return parameters used for Weight normalization.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("g", DynamicParam.ParamType.INT);
        return paramDefs;
    }

    /**
     * Sets parameters used for Weight Normalization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - g: g multiplier value for normalization. Default value 1.<br>
     *
     * @param params parameters used for weight normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("g")) g = params.getValueAsInteger("g");
    }

    /**
     * Resets Weight normalizer.
     *
     */
    public void reset() {
        Ws.clear();
        iNorms.clear();
    }

    /**
     * Sets flag for Weight normalization if neural network is in training state.
     *
     * @param isTraining if true neural network is in state otherwise false.
     */
    public void setTraining(boolean isTraining) {
        this.isTraining = isTraining;
    }

    /**
     * Sets optimizer for normalizer.
     *
     * @param optimizer optimizer
     */
    public void setOptimizer(Optimizer optimizer) {
    }

    /**
     * Defined which parameters are to be normalized.
     *
     * @param norm normalizable parameters.
     */
    public void setNormalizableParameters(HashSet<Matrix> norm) {
        this.norm = norm;
    }

    /**
     * Normalizes each weight for forward step i.e. multiplies each weight matrix by g / sqrt(2-norm of weights).
     *
     * @param W weight for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forward(Matrix W) throws MatrixException {
        if (!norm.contains(W)) return;
        Ws.put(W, W.copy());
        double iNorm = 1 / Math.sqrt(W.norm(2));
        iNorms.put(W, iNorm);
        W.multiply(g * iNorm, W);
    }

    /**
     * Finalizes forward step for normalization.<br>
     * Used typically for weight normalization.<br>
     *
     * @param W weight for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardFinalize(Matrix W) throws MatrixException {
        if (!norm.contains(W)) return;
        if (!isTraining) W.setEqualTo(Ws.get(W));
    }

    /**
     * Executes backward propagation step for Weight normalization.<br>
     * Calculates gradients backwards at step end for previous layer.<br>
     *
     * @param W weight for backward normalization.
     * @param dW gradient of weight for backward normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backward(Matrix W, Matrix dW) throws MatrixException {
        if (!norm.contains(W)) return;
        double iNorm = iNorms.get(W);
        double dg = dW.dot(W.T()).multiply(iNorm).sum();
        dW.multiply(g * iNorm).subtract(Ws.get(W).multiply(dg * g * Math.pow(iNorm, 2)), dW);
        W.setEqualTo(Ws.get(W));
    }

    /**
     * Not used.
     *
     * @param node node for normalization.
     */
    public void forward(Node node) {}

    /**
     * Not used.
     *
     * @param node node for normalization.
     */
    public void backward(Node node) {}

    /**
     * Not used.
     *
     * @param node node for normalization.
     * @param inputIndex input index for normalization.
     */
    public void forward(Node node, int inputIndex) {}

    /**
     * Not used.
     *
     * @param node node for normalization.
     * @param outputIndex input index for normalization.
     */
    public void backward(Node node, int outputIndex) {}

}