/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.optimization;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements vanilla Gradient Descent optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/ <br>
 *
 */
public class GradientDescent extends AbstractOptimizer {

    /**
     * Parameter name types for GradientDescent.
     *     - learningRate: learning rate for optimizer. Default value 0.01.<br>
     *
     */
    private final static String paramNameTypes = "(learningRate:DOUBLE)";

    /**
     * Learning rate for Gradient Descent. Default value 0.001.
     *
     */
    private double learningRate;

    /**
     * Default constructor for Gradient Descent.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public GradientDescent() throws DynamicParamException {
        super(OptimizationType.GRADIENT_DESCENT, GradientDescent.paramNameTypes);
    }

    /**
     * Constructor for Gradient Descent.
     *
     * @param params parameters for Gradient Descent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public GradientDescent(String params) throws DynamicParamException {
        super(OptimizationType.GRADIENT_DESCENT, GradientDescent.paramNameTypes, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        learningRate = 0.01;
    }

    /**
     * Sets parameters used for Gradient Descent.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 0.01.<br>
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

}

