/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.optimization;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunctionType;

import java.util.HashMap;

/**
 * Implements Adagrad optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/ <br>
 *
 */
public class Adagrad extends AbstractOptimizer {

    /**
     * Parameter name types for Adagrad.
     *     - learningRate: learning rate for optimizer. Default value 1.<br>
     *
     */
    private final static String paramNameTypes = "(learningRate:DOUBLE)";

    /**
     * Learning rate for Adagrad. Default value 0.01.
     *
     */
    private double learningRate;

    /**
     * Hash map to store gradients from previous steps.
     *
     */
    private final HashMap<Matrix, Matrix> m2Sum = new HashMap<>();

    /**
     * Default constructor for Adagrad.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Adagrad() throws DynamicParamException {
        super(OptimizationType.ADAGRAD, Adagrad.paramNameTypes);
    }

    /**
     * Constructor for Adagrad.
     *
     * @param params parameters for Adagrad.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Adagrad(String params) throws DynamicParamException {
        super(OptimizationType.ADAGRAD, Adagrad.paramNameTypes, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        learningRate = 0.01;
    }

    /**
     * Sets parameters used for Adagrad.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 0.01.<br>
     *
     * @param params parameters used for Adagrad.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("learningRate")) learningRate = params.getValueAsDouble("learningRate");
    }

    /**
     * Resets optimizer state.
     *
     */
    public void reset() {
        m2Sum.clear();
    }

    /**
     * Optimizes single matrix (M) using calculated matrix gradient (dM).<br>
     * Matrix can be for example weight or bias matrix with gradient.<br>
     *
     * @param matrix matrix to be optimized.
     * @param matrixGradient matrix gradients for optimization step.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void optimize(Matrix matrix, Matrix matrixGradient) throws MatrixException, DynamicParamException {
        Matrix dM2Sum = getParameterMatrix(m2Sum, matrix);

        dM2Sum.add(matrixGradient.multiply(matrixGradient), dM2Sum);

        double epsilon = 10E-8;
        matrix.subtract(matrixGradient.divide(dM2Sum.add(epsilon).apply(UnaryFunctionType.SQRT)).multiply(learningRate), matrix);
    }

}
