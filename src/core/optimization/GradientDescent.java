/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.optimization;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Matrix;
import utils.MatrixException;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Class that implements vanilla Gradient Descent optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/<br>
 *
 */
public class GradientDescent implements Optimizer, Serializable {

    private static final long serialVersionUID = 954492995414169438L;

    /**
     * Learning rate for Gradient Descent. Default value 0.001.
     *
     */
    private double learningRate = 0.001;

    /**
     * Relative size of mini batch.
     *
     */
    private double miniBatchFactor = 1;

    /**
     * Default constructor for Gradient Descent.
     *
     */
    public GradientDescent() {
    }

    /**
     * Constructor for Gradient Descent.
     *
     * @param params parameters for Gradient Descent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public GradientDescent(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Gets parameters used for Gradient Descent.
     *
     * @return parameters used for Gradient Descent.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("learningRate", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for Gradient Descent.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *
     * @param params parameters used for Gradient Descent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("learningRate")) learningRate = params.getValueAsDouble("learningRate");
    }

    /**
     * Resets optimizer state. Not relevant for Gradient Descent.
     *
     */
    public void reset() {}

    /**
     * Sets relative size of mini batch.
     *
     * @param miniBatchFactor relative size of mini batch.
     */
    public void setMiniBatchFactor(double miniBatchFactor) {
        this.miniBatchFactor = miniBatchFactor;
    }

    /**
     * Optimizes given weight (W) and bias (B) pair with given gradients respectively.
     *
     * @param W weight matrix to be optimized.
     * @param dW weight gradients for optimization step.
     * @param B bias matrix to be optimized.
     * @param dB bias gradients for optimization step.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void optimize(Matrix W, Matrix dW, Matrix B, Matrix dB) throws MatrixException {
        optimize(W, dW);
        optimize(B, dB);
    }

    /**
     * Optimizes single matrix (M) using calculated matrix gradient (dM).<br>
     * Matrix can be for example weight or bias matrix with gradient.<br>
     *
     * @param M matrix to be optimized.
     * @param dM matrix gradients for optimization step.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void optimize(Matrix M, Matrix dM) throws MatrixException {
        M.subtract(dM.multiply(learningRate * miniBatchFactor), M);
    }

}

