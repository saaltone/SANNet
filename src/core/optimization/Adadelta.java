/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.optimization;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunctionType;

import java.util.HashMap;

/**
 * Implements Adadelta optimizer.<br>
 * <br>
 * Reference: <a href="http://ruder.io/optimizing-gradient-descent/">...</a> <br>
 *
 */
public class Adadelta extends AbstractOptimizer {

    /**
     * Parameter name types for Adadelta.
     *     - learningRate: learning rate for optimizer. Default value 1.<br>
     *     - gamma: gamma value for optimizer. Default value 0.95.<br>
     *
     */
    private final static String paramNameTypes = "(learningRate:DOUBLE), " +
            "(gamma:DOUBLE)";

    /**
     * Learning rate for Adadelta. Default value 1.
     *
     */
    private double learningRate;

    /**
     * Gamma term for Adadelta. Default value 0.95.
     *
     */
    private double gamma;

    /**
     * Hash map to store gradients from previous steps.
     *
     */
    private final HashMap<Matrix, Matrix> eg2 = new HashMap<>();

    /**
     * Hash map to store gradient deltas from previous steps.
     *
     */
    private final HashMap<Matrix, Matrix> ed2 = new HashMap<>();

    /**
     * Default constructor for Adadelta.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Adadelta() throws DynamicParamException {
        super(OptimizationType.ADADELTA, Adadelta.paramNameTypes);
    }

    /**
     * Constructor for Adadelta.
     *
     * @param params parameters for Adadelta.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Adadelta(String params) throws DynamicParamException {
        super(OptimizationType.ADADELTA, Adadelta.paramNameTypes, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        learningRate = 1;
        gamma = 0.95;
    }

    /**
     * Sets parameters used for Adadelta.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 1.<br>
     *     - gamma: gamma value for optimizer. Default value 0.95.<br>
     *
     * @param params parameters used for Adadelta.
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
        ed2.clear();
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
        mEg2 = mEg2.multiply(gamma).add(matrixGradient.power(2).multiply(1 - gamma));
        setParameterMatrix(eg2, matrix, mEg2);

        final double epsilon = 10E-8;
        Matrix mEd2 = getParameterMatrix(ed2, matrix);
        Matrix Ed = mEd2.add(epsilon).apply(UnaryFunctionType.SQRT).divide(mEg2.add(epsilon).apply(UnaryFunctionType.SQRT)).multiply(matrixGradient);
        matrix.subtractBy(Ed.multiply(learningRate));

        mEd2 = mEd2.multiply(gamma).add(Ed.power(2).multiply(1 - gamma));
        setParameterMatrix(ed2, matrix, mEd2);
    }

}
