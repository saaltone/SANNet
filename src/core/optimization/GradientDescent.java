/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.optimization;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Class that implements vanilla Gradient Descent optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/ <br>
 *
 */
public class GradientDescent implements Optimizer, Serializable {

    private static final long serialVersionUID = 954492995414169438L;

    /**
     * Optimization type.
     *
     */
    private final OptimizationType optimizationType = OptimizationType.GRADIENT_DESCENT;

    /**
     * Learning rate for Gradient Descent. Default value 0.001.
     *
     */
    private double learningRate = 0.001;

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
     * Returns parameters used for Gradient Descent.
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
     * Optimizes given weight (W) and bias (B) pair with given gradients respectively.
     *
     * @param weight weight matrix to be optimized.
     * @param weightGradient weight gradients for optimization step.
     * @param bias bias matrix to be optimized.
     * @param biasGradient bias gradients for optimization step.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void optimize(Matrix weight, Matrix weightGradient, Matrix bias, Matrix biasGradient) throws MatrixException {
        optimize(weight, weightGradient);
        optimize(bias, biasGradient);
    }

    /**
     * Optimizes single matrix (M) using calculated matrix gradient (dM).<br>
     * Matrix can be for example weight or bias matrix with gradient.<br>
     *
     * @param matrix matrix to be optimized.
     * @param matrixGradient matrix gradients for optimization step.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void optimize(Matrix matrix, Matrix matrixGradient) throws MatrixException {
        matrix.subtract(matrixGradient.multiply(learningRate), matrix);
    }

    /**
     * Returns name of optimizer.
     *
     * @return name of optimizer.
     */
    public String getName() {
        return optimizationType.toString();
    }

}

