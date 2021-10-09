/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.regularization;

import utils.Configurable;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Sequence;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunctionType;

import java.io.Serial;
import java.io.Serializable;

/**
 * Implements L1 (lasso) regularization.<br>
 * <br>
 * Reference: https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c<br>
 *
 */
public class L1_Regularization implements Configurable, Regularization, Serializable {

    @Serial
    private static final long serialVersionUID = -7323953827581797724L;

    /**
     * Parameter name types for L1_Regularization.
     *     - lambda: lambda value for regularization. Default value: 0.01.<br>
     *
     */
    private final static String paramNameTypes = "(lambda:DOUBLE)";

    /**
     * Type of regularization.
     *
     */
    private final RegularizationType regularizationType = RegularizationType.L1_REGULARIZATION;

    /**
     * Regularization rate.
     *
     */
    private double lambda;

    /**
     * Constructor for L1 regularization class.
     *
     */
    public L1_Regularization() {
        initializeDefaultParams();
    }

    /**
     * Constructor for L1 regularization class.
     *
     * @param params parameters for L1 regularization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public L1_Regularization(String params) throws DynamicParamException {
        this();
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        lambda = 0.01;
    }

    /**
     * Returns parameters used for L1 regularization.
     *
     * @return parameters used for L1 regularization.
     */
    public String getParamDefs() {
        return L1_Regularization.paramNameTypes;
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
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return cumulated error from L1 regularization.
     */
    public double error(Matrix weight) throws DynamicParamException, MatrixException {
        return lambda * weight.apply(UnaryFunctionType.ABS).sum();
    }

    /**
     * Regulates weights by calculating 1- norm of weights and adding it to weight gradient sum.
     *
     * @param weight weight matrix.
     * @param weightGradientSum gradient sum of weight.
     */
    public void backward(Matrix weight, Matrix weightGradientSum) throws MatrixException {
        weightGradientSum.add(weight.apply((value) -> lambda * Math.signum(value)), weightGradientSum);
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
