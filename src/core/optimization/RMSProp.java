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
 * Class that implements RMSProp optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/<br>
 *
 */
public class RMSProp implements Optimizer, Serializable {

    private static final long serialVersionUID = 3251200097077919746L;

    /**
     * Optimization type.
     *
     */
    private final OptimizationType optimizationType;

    /**
     * Learning rate for RMSProp. Default value 0.001.
     *
     */
    private double learningRate = 0.001;

    /**
     * Gamma term for RMSProp. Default value 0.9.
     *
     */
    private double gamma = 0.9;

    /**
     * Hash map to store gradients from previous steps.
     *
     */
    private transient HashMap<Matrix, Matrix> eg2;

    /**
     * Default constructor for RMSProp.
     *
     * @param optimizationType optimizationType.
     */
    public RMSProp(OptimizationType optimizationType) {
        this.optimizationType = optimizationType;
    }

    /**
     * Constructor for RMSProp.
     *
     * @param optimizationType optimizationType.
     * @param params parameters for RMSProp.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public RMSProp(OptimizationType optimizationType, String params) throws DynamicParamException {
        this(optimizationType);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for RMSProp.
     *
     * @return parameters used for RMSProp.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("learningRate", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("gamma", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
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
        eg2 = new HashMap<>();
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
        if (eg2 == null) eg2 = new HashMap<>();

        Matrix mEg2;
        if (eg2.containsKey(matrix)) mEg2 = eg2.get(matrix);
        else eg2.put(matrix, mEg2 = new DMatrix(matrix.getRows(), matrix.getColumns()));

        eg2.put(matrix, mEg2 = mEg2.multiply(gamma).add(matrixGradient.power(2).multiply(1 - gamma)));

        double epsilon = 10E-8;
        matrix.subtract(matrixGradient.divide(mEg2.add(epsilon).apply(UnaryFunctionType.SQRT)).multiply(learningRate), matrix);
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

