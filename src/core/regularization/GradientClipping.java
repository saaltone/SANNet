/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.regularization;

import core.layer.Connector;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Matrix;
import utils.MatrixException;

import java.io.Serializable;
import java.util.HashMap;
import java.util.TreeMap;

/**
 * Implements gradient clipping class.<br>
 * Gradient clipping cuts gradient when certain threshold is reached to prevent then from growing too big i.e. exploding.<br>
 * <br>
 * Reference: https://hackernoon.com/gradient-clipping-57f04f0adae<br>
 *
 */
public class GradientClipping implements Regularization, Serializable {

    private static final long serialVersionUID = -2462517110247269075L;

    /**
     * Reference to connector between previous and next layer.
     *
     */
    private final Connector connector;

    /**
     * Threshold for gradient clipping.
     *
     */
    private double threshold = 0.1;

    /**
     * Constructor for gradient clipping class.
     *
     * @param connector reference to connector between previous and next layer.
     * @param toHiddenLayer true if next layer if hidden layer otherwise false.
     */
    public GradientClipping(Connector connector, boolean toHiddenLayer) {
        this.connector = connector;
    }

    /**
     * Constructor for gradient clipping class.
     *
     * @param connector reference to connector between previous and next layer.
     * @param toHiddenLayer true if next layer if hidden layer otherwise false.
     * @param params parameters for gradient clipping.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public GradientClipping(Connector connector, boolean toHiddenLayer, String params) throws DynamicParamException {
        this(connector, toHiddenLayer);
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Gets parameters used for gradient clipping.
     *
     * @return parameters used for gradient clipping.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("threshold", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for gradient clipping.<br>
     * <br>
     * Supported parameters are:<br>
     *     - threshold: threshold for clipping gradients. Default value 0.1.<br>
     *
     * @param params parameters used for gradient clipping.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("threshold")) threshold = params.getValueAsDouble("threshold");
    }

    /**
     * Not used.
     *
     */
    public void reset() {}

    /**
     * Not used.
     *
     * @param isTraining if true neural network is in state otherwise false.
     */
    public void setTraining(boolean isTraining) {
    }

    /**
     * Not used.
     *
     * @param ins input samples for forward step.
     * @param index if index is zero or positive value operation is executed for this sample. if index is -1 operation is executed for all samples.
     */
    public void forwardPre(TreeMap<Integer, Matrix> ins, int index) {}

    /**
     * Not used.
     *
     * @param outs output samples for forward step.
     */
    public void forwardPost(TreeMap<Integer, Matrix> outs) {}

    /**
     * Not used.
     *
     * @return not used.
     */
    public double error() {
        return 0;
    }

    /**
     * Not used.
     *
     * @param index if index is zero or positive value operation is executed for this sample. if index is -1 operation is executed for all samples.
     */
    public void backward(int index) {}

    /**
     * Executes gradient clipping prior weight update step for neural network.<br>
     * Compares gradient sum against specific threshold and cuts gradients proportionally is threshold is exceeded.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void update() throws MatrixException {
        for (Matrix W : connector.getReg()) {
            Matrix dWSum = connector.getdWsSums(W);
            double dWSum_l2norm = Math.sqrt(dWSum.norm(2));
            if (dWSum_l2norm > threshold) dWSum.multiply(threshold / dWSum_l2norm, dWSum);
        }
    }

}
