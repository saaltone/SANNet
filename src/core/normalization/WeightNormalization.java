/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.normalization;

import core.layer.Connector;
import utils.*;

import java.io.Serializable;
import java.util.HashMap;
import java.util.TreeMap;

/**
 * Class that implements Weight Normalization for neural network layer.<br>
 * <br>
 * Reference: https://arxiv.org/pdf/1602.07868.pdf<br>
 *
 */
public class WeightNormalization implements Normalization, Serializable {

    private static final long serialVersionUID = 1741544680542755148L;

    /**
     * Reference to connector between previous and next layer.
     *
     */
    private Connector connector;

    /**
     * Indicates to regularizer or normalizer if neural network is in training mode.
     *
     */
    private transient boolean isTraining;

    /**
     * Tree map for un-normalized weights.
     *
     */
    private transient HashMap<Matrix, Matrix> Ws;

    /**
     * Tree map for storing weight normalizing factors (1 / sqrt(2-norm W))
     *
     */
    private transient HashMap<Matrix, Double> iNorms;

    /**
     * Weight normalization scalar.
     *
     */
    private double g = 1;

    /**
     * Constructor for weight normalization class.
     *
     * @param connector reference to connector between previous and next layer.
     */
    public WeightNormalization(Connector connector) {
        this.connector = connector;
    }

    /**
     * Constructor for weight normalization class.
     *
     * @param connector reference to connector between previous and next layer.
     * @param params parameters for weight normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public WeightNormalization(Connector connector, String params) throws DynamicParamException {
        this(connector);
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Gets parameters used for weight normalization.
     *
     * @return parameters used for weight normalization.
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
     * Resets weight normalizer.
     *
     */
    public void reset() {
        Ws = new HashMap<>();
        iNorms = new HashMap<>();
    }

    /**
     * Sets flag for weight normalization if neural network is in training state.
     *
     * @param isTraining if true neural network is in state otherwise false.
     */
    public void setTraining(boolean isTraining) {
        this.isTraining = isTraining;
    }

    /**
     * Normalizes each weight for forward ste i.e. multiplies each weight matrix by g / sqrt(2-norm of weights).
     *
     * @param ins input samples for forward step.
     * @param channels not used. Only relevant for convolutional layer.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardPre(TreeMap<Integer, Matrix> ins, int channels) throws MatrixException {
        reset();
        for (Matrix W : connector.getReg()) {
            Ws.put(W, W.copy());
            double iNorm = 1 / Math.sqrt(W.norm(2));
            iNorms.put(W, iNorm);
            W.multiply(g * iNorm, W);
        }
    }

    /**
     * Restores original weight matrices after forward step.
     *
     * @param outs output samples for forward step.
     */
    public void forwardPost(TreeMap<Integer, Matrix> outs) throws MatrixException {
        for (Matrix W : connector.getReg()) {
            W.equal(Ws.get(W));
        }
    }

    /**
     * Executes backward propagation step for weight normalization.<br>
     * Calculates gradients backwards at step end for previous layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backward() throws MatrixException {
        for (Matrix W : connector.getReg()) {
            double iNorm = iNorms.get(W);
            TreeMap<Integer, Matrix> dWs = connector.getdWs().get(W);
            for (Integer index1 : dWs.keySet()) {
                Matrix dW = dWs.get(index1);
                Matrix dg = dW.multiply(W).multiply(iNorm);
                Matrix dWsP = dW.multiply(g * iNorm).subtract(dg.multiply(g * Math.pow(iNorm, 2)).multiply(W));
                dWs.put(index1, dWsP);
            }
        }
    }

}