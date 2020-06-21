/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.optimization;

import utils.*;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunctionType;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Class that implements Adagrad optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/<br>
 *
 */
public class Adagrad implements Optimizer, Serializable {

    private static final long serialVersionUID = -8831643329108200212L;

    /**
     * Optimization type.
     *
     */
    private final OptimizationType optimizationType;

    /**
     * Learning rate for Adagrad. Default value 0.01.
     *
     */
    private double learningRate = 0.01;

    /**
     * Hash map to store gradients from previous steps.
     *
     */
    private transient HashMap<Matrix, Matrix> m2Sum;

    /**
     * Default constructor for Adagrad.
     *
     * @param optimizationType optimizationType.
     */
    public Adagrad(OptimizationType optimizationType) {
        this.optimizationType = optimizationType;
    }

    /**
     * Constructor for Adagrad.
     *
     * @param optimizationType optimizationType.
     * @param params parameters for Adagrad.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Adagrad(OptimizationType optimizationType, String params) throws DynamicParamException {
        this(optimizationType);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for Adagrad.
     *
     * @return parameters used for Adagrad.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("learningRate", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
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
        m2Sum = new HashMap<>();
    }

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
        if (m2Sum == null) m2Sum = new HashMap<>();

        Matrix dM2Sum;
        if (m2Sum.containsKey(matrix)) dM2Sum = m2Sum.get(matrix);
        else m2Sum.put(matrix, dM2Sum = new DMatrix(matrix.getRows(), matrix.getColumns()));

        dM2Sum.add(matrixGradient.multiply(matrixGradient), dM2Sum);

        double epsilon = 10E-8;
        matrix.subtract(matrixGradient.divide(dM2Sum.add(epsilon).apply(UnaryFunctionType.SQRT)).multiply(learningRate), matrix);
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
