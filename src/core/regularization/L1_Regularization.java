/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.regularization;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Sequence;
import utils.matrix.MMatrix;
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
     * Type of regularization.
     *
     */
    private final RegularizationType regularizationType;

    /**
     * Regularization rate.
     *
     */
    private double lambda = 0.01;

    /**
     * Constructor for L1 regularization class.
     *
     * @param regularizationType regularizationType.
     */
    public L1_Regularization(RegularizationType regularizationType) {
        this.regularizationType = regularizationType;
    }

    /**
     * Constructor for L1 regularization class.
     *
     * @param regularizationType regularizationType.
     * @param params parameters for L1 regularization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public L1_Regularization(RegularizationType regularizationType, String params) throws DynamicParamException {
        this(regularizationType);
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
    public void setTraining(boolean isTraining) {
    }

    /**
     * Not used.
     *
     * @param sequence input sequence.
     */
    public void forward(Sequence sequence) {}

    /**
     * Not used.
     *
     * @param inputs inputs.
     */
    public void forward(MMatrix inputs) {}

    /**
     * Calculates and returns cumulated error from L1 regularization.
     * This is added to the total output error of neural network.
     *
     * @param weight weight matrix.
     * @return cumulated error from L1 regularization.
     */
    public double error(Matrix weight) {
        return lambda * weight.norm(1);
    }

    /**
     * Regulates weights by calculating 1- norm of weights and adding it to weight gradient sum.
     *
     * @param weight weight matrix.
     * @param weightGradientSum gradient sum of weight.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backward(Matrix weight, Matrix weightGradientSum) throws MatrixException {
        weightGradientSum.add(weight.apply((value) -> lambda * Math.abs(value)), weightGradientSum);
    }

    /**
     * Returns name of regularization.
     *
     * @return name of regularization.
     */
    public String getName() {
        return regularizationType.toString();
    }

}
