/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.optimization;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunctionType;

import java.util.HashMap;

/**
 * Implements RMSProp optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/ <br>
 *
 */
public class RMSProp extends AbstractOptimizer {

    /**
     * Parameter name types for RMSProp.
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - gamma: gamma value for optimizer. Default value 0.9.<br>
     *
     */
    private final static String paramNameTypes = "(learningRate:DOUBLE), " +
            "(gamma:DOUBLE)";

    /**
     * Learning rate for RMSProp. Default value 0.001.
     *
     */
    private double learningRate;

    /**
     * Gamma term for RMSProp. Default value 0.9.
     *
     */
    private double gamma;

    /**
     * Hash map to store gradients from previous steps.
     *
     */
    private final HashMap<Matrix, Matrix> eg2 = new HashMap<>();

    /**
     * Default constructor for RMSProp.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public RMSProp() throws DynamicParamException {
        super(OptimizationType.RMSPROP, RMSProp.paramNameTypes);
    }

    /**
     * Constructor for RMSProp.
     *
     * @param params parameters for RMSProp.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public RMSProp(String params) throws DynamicParamException {
        super(OptimizationType.RMSPROP, RMSProp.paramNameTypes, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        learningRate = 0.001;
        gamma = 0.9;
    }

    /**
     * Sets parameters used for RMSProp.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - gamma: gamma value for optimizer. Default value 0.9.<br>
     *
     * @param params parameters used for RMSProp.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("learningRate")) learningRate = params.getValueAsDouble("learningRate");
        if (params.hasParam("gamma")) gamma = params.getValueAsDouble("gamma");
    }

    /**
     * Resets optimizer state.
     *
     */
    public void reset() {
        eg2.clear();
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
        Matrix mEg2 = getParameterMatrix(eg2, matrix);

        eg2.put(matrix, mEg2 = mEg2.multiply(gamma).add(matrixGradient.power(2).multiply(1 - gamma)));

        double epsilon = 10E-8;
        matrix.subtract(matrixGradient.divide(mEg2.add(epsilon).apply(UnaryFunctionType.SQRT)).multiply(learningRate), matrix);
    }

}

