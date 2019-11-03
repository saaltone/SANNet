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
 * Implements L1 (lasso) regularization.<br>
 * <br>
 * Reference: https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c<br>
 *
 */
public class L1_Regularization implements Regularization, Serializable {

    private static final long serialVersionUID = -7323953827581797724L;

    /**
     * Regularization rate.
     *
     */
    private double lambda = 0.01;

    /**
     * Constructor for L1 regularization class.
     *
     */
    public L1_Regularization() {
    }

    /**
     * Constructor for L1 regularization class.
     *
     * @param params parameters for L1 regularization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public L1_Regularization(String params) throws DynamicParamException {
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for L1 regularization.
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
     * @param sequence input sequence.
     */
    public void forward(Sequence sequence) {

    }

    /**
     * Not used.
     *
     * @param W weight matrix.
     */
    public void forward(Matrix W) {}

    /**
     * Calculates and returns cumulated error from L1 regularization.
     * This is added to the total output error of neural network.
     *
     * @param W weight matrix.
     * @return cumulated error from L1 regularization.
     */
    public double error(Matrix W) {
        return lambda * W.norm(1);
    }

    /**
     * Regulates weights by calculating 1- norm of weights and adding it to weight gradient sum.
     *
     * @param W weight matrix.
     * @param dWSum gradient sum of weight.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backward(Matrix W, Matrix dWSum) throws MatrixException {
        Matrix.MatrixUnaryOperation function = (value) -> lambda * Math.abs(value);
        dWSum.add(W.apply(function), dWSum);
    }

}
