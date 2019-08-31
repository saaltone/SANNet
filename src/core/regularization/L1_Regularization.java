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
 * Implements L1 (lasso) regularization.<br>
 * <br>
 * Reference: https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c<br>
 *
 */
public class L1_Regularization implements Regularization, Serializable {

    private static final long serialVersionUID = -7323953827581797724L;

    /**
     * Reference to connector between previous and next layer.
     *
     */
    private final Connector connector;

    /**
     * Regularization rate.
     *
     */
    private double lambda = 0.01;

    /**
     * Constructor for L1 regularization class.
     *
     * @param connector reference to connector between previous and next layer.
     * @param toHiddenLayer true if next layer if hidden layer otherwise false.
     */
    public L1_Regularization(Connector connector, boolean toHiddenLayer) {
        this.connector = connector;
    }

    /**
     * Constructor for L1 regularization class.
     *
     * @param connector reference to connector between previous and next layer.
     * @param toHiddenLayer true if next layer if hidden layer otherwise false.
     * @param params parameters for L1 regularization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public L1_Regularization(Connector connector, boolean toHiddenLayer, String params) throws DynamicParamException {
        this(connector, toHiddenLayer);
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Gets parameters used for L1 regularization.
     *
     * @return parameters used for L1 regularization.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("lambda", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for L1 regularization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - lambda: lambda value for regularization. Default value: 0.01.<br>
     *
     * @param params parameters used for L1 regularization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("lambda")) lambda = params.getValueAsDouble("lambda");
    }

    /**
     * Not used.
     *
     */
    public void reset() {}

    /**
     * Not used.
     *
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
     * Calculates and returns cumulated error from L1 regularization.
     * This is added to the total output error of neural network.
     *
     * @return cumulated error from L1 regularization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double error() throws MatrixException {
        double error = 0;
        for (Matrix W : connector.getReg()) {
            error += lambda * W.norm(1);
        }
        return error;
    }

    /**
     * Not used.
     *
     * @param index if index is zero or positive value operation is executed for this sample. if index is -1 operation is executed for all samples.
     */
    public void backward(int index) {
    }

    /**
     * Regulates weights by calculating 1- norm of weights and adding it to weight gradient sum.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void update() throws MatrixException {
        Matrix.MatrixUniOperation function = (value) -> lambda * Math.abs(value);
        for (Matrix W : connector.getReg()) {
            Matrix dWSum = connector.getdWsSums(W);
            dWSum.add(W.apply(function), dWSum);
        }
    }

}
