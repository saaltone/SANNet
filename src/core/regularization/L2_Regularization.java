/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.regularization;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Sequence;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Implements L2 (ridge) regularization.<br>
 * <br>
 * Reference: https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c<br>
 *
 */
public class L2_Regularization implements Regularization, Serializable {

    private static final long serialVersionUID = 7179599386737519841L;

    /**
     * Regularization rate.
     *
     */
    private double lambda = 0.01;

    /**
     * Constructor for L2 regularization class.
     *
     */
    public L2_Regularization() {
    }

    /**
     * Constructor for L2 regularization class.
     *
     * @param params parameters for L2 regularization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public L2_Regularization(String params) throws DynamicParamException {
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for L2 regularization.
     *
     * @return parameters used for L2 regularization.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("lambda", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for L2 regularization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - lambda: lambda value for regularization. Default value: 0.01.<br>
     *
     * @param params parameters used for L2 regularization.
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
     * @param sequence input sequence.
     */
    public void forward(Sequence sequence) {

    }

    /**
     * Not used.
     *
     */
    public void setTraining(boolean isTraining) {
    }

    /**
     * Not used.
     *
     * @param W weight matrix.
     */
    public void forward(Matrix W) {}

    /**
     * Calculates and returns cumulated error from L2 regularization.<br>
     * This is added to the total output error of neural network.<br>
     *
     * @param W weight matrix.
     * @return cumulated error from L2 regularization.
     */
    public double error(Matrix W) {
        return lambda * W.norm(2);
    }

    /**
     * Regulates weights by calculating 2- norm of weights and adding it to weight gradient sum.
     *
     * @param W weight matrix.
     * @param dWSum gradient sum of weight.
     */
    public void backward(Matrix W, Matrix dWSum) throws MatrixException {
        dWSum.add(W.multiply(2 * lambda), dWSum);
    }

}

