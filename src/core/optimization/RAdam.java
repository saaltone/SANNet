/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.optimization;

import utils.*;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunctionType;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Class that implements Rectified Adam optimizer.<br>
 * <br>
 * Reference: https://arxiv.org/abs/1908.03265<br>
 *
 */
public class RAdam implements Optimizer, Serializable {

    private static final long serialVersionUID = -2717951798872633802L;

    /**
     * Optimization type.
     *
     */
    private final OptimizationType optimizationType = OptimizationType.RADAM;

    /**
     * Learning rate for RAdam. Default value 0.010.
     *
     */
    private double learningRate = 0.001;

    /**
     * Beta1 term for RAdam. Default value 0.9.
     *
     */
    private double beta1 = 0.9;

    /**
     * Beta2 term for RAdam. Default value 0.999.
     *
     */
    private double beta2 = 0.999;

    /**
     * Hash map to store iteration counts.
     *
     */
    private HashMap<Matrix, Integer> iterations;

    /**
     * Maximum length of approximated SMA.
     *
     */
    private double pinf = 2 / (1 - beta2) - 1;

    /**
     * Hash map to store first moments (means).
     *
     */
    private HashMap<Matrix, Matrix> m;

    /**
     * Hash map to store second moments (uncentered variances).
     *
     */
    private HashMap<Matrix, Matrix> v;

    /**
     * Default constructor for RAdam.
     *
     */
    public RAdam() {
    }

    /**
     * Constructor for RAdam.
     *
     * @param params parameters for RAdam.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public RAdam(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for RAdam.
     *
     * @return parameters used for RAdam.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("learningRate", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("beta1", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("beta2", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for RAdam.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - beta1: beta1 value for optimizer. Default value 0.9.<br>
     *     - beta2: beta2 value for optimizer. Default value 0.999.<br>
     *
     * @param params parameters used for RAdam.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("learningRate")) learningRate = params.getValueAsDouble("learningRate");
        if (params.hasParam("beta1")) beta1 = params.getValueAsDouble("beta1");
        if (params.hasParam("beta2")) beta2 = params.getValueAsDouble("beta2");
        pinf = 2 / (1 - beta2) - 1;
    }

    /**
     * Resets optimizer state.
     *
     */
    public void reset() {
        iterations = new HashMap<>();
        m = new HashMap<>();
        v = new HashMap<>();
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
        if (iterations == null) iterations = new HashMap<>();
        if (m == null) m = new HashMap<>();
        if (v == null) v = new HashMap<>();

        int iteration;
        if (iterations.containsKey(matrix)) iterations.put(matrix, iteration = iterations.get(matrix) + 1);
        else iterations.put(matrix, iteration = 1);

        Matrix vM = null;
        if (v.containsKey(matrix)) vM = v.get(matrix);

        Matrix mM = null;
        if (m.containsKey(matrix)) mM = m.get(matrix);

        vM = vM == null ? matrixGradient.power(2).multiply(1 - beta2) : vM.multiply(beta2).add(matrixGradient.power(2).multiply(1 - beta2));
        v.put(matrix, vM);

        mM = mM == null ? matrixGradient.multiply(1 - beta1) : mM.multiply(beta1).add(matrixGradient.multiply(1 - beta1));
        m.put(matrix, mM);

        double beta1Iteration = Math.pow(beta1, iteration);
        double beta2Iteration = Math.pow(beta2, iteration);

        Matrix mMhat = mM.divide(1 - beta1Iteration);

        double stepSize = learningRate;
        double pt = pinf - 2 * iteration * beta2Iteration / (1 - beta2Iteration);
        if (pt > 4) {
            stepSize *=  Math.sqrt((1 - beta2Iteration) * ((pt - 4) * (pt - 2) * pinf) / ((pinf - 4) * (pinf - 2) * pt));
            double epsilon = 10E-8;
            matrix.subtract(mMhat.divide(vM.apply(UnaryFunctionType.SQRT).add(epsilon)).multiply(stepSize), matrix);
        }
        else {
            matrix.subtract(mMhat.multiply(stepSize), matrix);
        }
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

