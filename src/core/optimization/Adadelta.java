/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.optimization;

import utils.*;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunctionType;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashMap;

/**
 * Class that implements Adadelta optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/ <br>
 *
 */
public class Adadelta implements Configurable, Optimizer, Serializable {

    @Serial
    private static final long serialVersionUID = 1620048040058081811L;

    /**
     * Parameter name types for Adadelta.
     *     - learningRate: learning rate for optimizer. Default value 1.<br>
     *     - gamma: gamma value for optimizer. Default value 0.95.<br>
     *
     */
    private final static String paramNameTypes = "(learningRate:DOUBLE), " +
            "(gamma:DOUBLE)";

    /**
     * Parameters of optimizer.
     *
     */
    private final String params;

    /**
     * Optimization type.
     *
     */
    private final OptimizationType optimizationType = OptimizationType.ADADELTA;

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
    private transient HashMap<Matrix, Matrix> eg2;

    /**
     * Hash map to store gradient deltas from previous steps.
     *
     */
    private transient HashMap<Matrix, Matrix> ed2;

    /**
     * Default constructor for Adadelta.
     *
     */
    public Adadelta() {
        initializeDefaultParams();
        params = null;
    }

    /**
     * Constructor for Adadelta.
     *
     * @param params parameters for Adadelta.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Adadelta(String params) throws DynamicParamException {
        initializeDefaultParams();
        this.params = params;
        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
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
     * Returns parameters of optimizer.
     *
     * @return parameters for optimizer.
     */
    public String getParams() {
        return params;
    }

    /**
     * Returns parameters used for Adadelta.
     *
     * @return parameters used for Adadelta.
     */
    public String getParamDefs() {
        return Adadelta.paramNameTypes;
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
        eg2 = new HashMap<>();
        ed2 = new HashMap<>();
    }

    /**
     * Optimizes given weight (W) and bias (B) pair with given gradients respectively.
     *
     * @param weight weight matrix to be optimized.
     * @param weightGradient weight gradients for optimization step.
     * @param bias bias matrix to be optimized.
     * @param biasGradient bias gradients for optimization step.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void optimize(Matrix weight, Matrix weightGradient, Matrix bias, Matrix biasGradient) throws MatrixException, DynamicParamException {
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
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void optimize(Matrix matrix, Matrix matrixGradient) throws MatrixException, DynamicParamException {
        if (eg2 == null) eg2 = new HashMap<>();
        Matrix mEg2;
        if (eg2.containsKey(matrix)) mEg2 = eg2.get(matrix);
        else eg2.put(matrix, mEg2 = new DMatrix(matrix.getRows(), matrix.getColumns()));

        if (ed2 == null) ed2 = new HashMap<>();
        Matrix mEd2;
        if (ed2.containsKey(matrix)) mEd2 = ed2.get(matrix);
        else ed2.put(matrix, mEd2 = new DMatrix(matrix.getRows(), matrix.getColumns()));

        mEg2 = mEg2.multiply(gamma).add(matrixGradient.power(2).multiply(1 - gamma));

        double epsilon = 10E-8;
        Matrix Ed = mEd2.add(epsilon).apply(UnaryFunctionType.SQRT).divide(mEg2.add(epsilon).apply(UnaryFunctionType.SQRT)).multiply(matrixGradient);
        matrix.subtract(Ed.multiply(learningRate), matrix);
        mEd2 = mEd2.multiply(gamma).add(Ed.power(2).multiply(1 - gamma));

        eg2.put(matrix, mEg2);
        ed2.put(matrix, mEd2);
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
