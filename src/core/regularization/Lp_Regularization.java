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

import java.io.Serial;
import java.io.Serializable;

/**
 * Implements Lp regularization (experimental). P here is any norm higher or equal to 1.<br>
 * Setting p = 1 this becomes L1 regularization and setting p = 2 this becomes L2 regularization.<br>
 * <br>
 * This is experimental regularization method.<br>
 *
 */
public class Lp_Regularization implements Configurable, Regularization, Serializable {

    @Serial
    private static final long serialVersionUID = -7833984930510523396L;

    /**
     * Parameter name types for Lp_Regularization.
     *     - lambda: lambda value for regularization. Default value: 0.01.<br>
     *     - p: p norm of normalizer. Default 3.<br>
     *
     */
    private final static String paramNameTypes = "(lambda:DOUBLE), " +
            "(p:INT)";

    /**
     * Type of regularization.
     *
     */
    private final RegularizationType regularizationType = RegularizationType.LP_REGULARIZATION;

    /**
     * Regularization rate.
     *
     */
    private double lambda;

    /**
     * Order of norm.
     *
     */
    private int p;

    /**
     * Constructor for Lp regularization class.
     *
     */
    public Lp_Regularization() {
        initializeDefaultParams();
    }

    /**
     * Constructor for Lp regularization class.
     *
     * @param params parameters for Lp regularization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Lp_Regularization(String params) throws DynamicParamException {
        this();
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        lambda = 0.01;
        p = 3;
    }

    /**
     * Returns parameters used for Lp regularization.
     *
     * @return parameters used for Lp regularization.
     */
    public String getParamDefs() {
        return Lp_Regularization.paramNameTypes;
    }

    /**
     * Sets parameters used for Lp regularization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - lambda: lambda value for regularization. Default value: 0.01.<br>
     *     - p: p norm of normalizer. Default 3.<br>
     *
     * @param params parameters used for Lp regularization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("lambda")) lambda = params.getValueAsDouble("lambda");
        if (params.hasParam("p")) p = params.getValueAsInteger("p");
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
     * Calculates and returns cumulated error from Lp regularization.<br>
     * This is added to the total output error of neural network.<br>
     *
     * @param weight weight matrix.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return cumulated error from L2 regularization.
     */
    public double error(Matrix weight) throws DynamicParamException, MatrixException {
        return lambda * weight.power(p).sum();
    }

    /**
     * Regulates weights by calculating p- norm of weights and adding it to weight gradient sum.
     *
     * @param weight weight matrix.
     * @param weightGradientSum gradient sum of weight.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backward(Matrix weight, Matrix weightGradientSum) throws MatrixException {
        weightGradientSum.add(weight.apply((value) -> value != 0 ? p * lambda * Math.pow(Math.abs(value), p - 1) / value : 0), weightGradientSum);
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
