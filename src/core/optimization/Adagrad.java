/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.optimization;

import utils.*;

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
     * Relative size of mini batch.
     *
     */
    private double miniBatchFactor = 1;

    /**
     * Default constructor for Adagrad.
     *
     */
    public Adagrad() {
    }

    /**
     * Constructor for Adagrad.
     *
     * @param params parameters for Adagrad.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Adagrad(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Gets parameters used for Adagrad.
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
        if (m2Sum == null) m2Sum = new HashMap<>();
        Matrix dM2Sum;
        if (m2Sum.containsKey(M)) dM2Sum = m2Sum.get(M);
        else m2Sum.put(M, dM2Sum = new DMatrix(M.getRows(), M.getCols()));

        dM2Sum.add(dM.multiply(dM), dM2Sum);

        /**
         * Epsilon term for Adagrad. Default value 10E-8.<br>
         * Term provides mathematical stability for optimizer.<br>
         *
         */
        double epsilon = 10E-8;
        M.subtract(dM.multiply(dM2Sum.add(epsilon).apply(UniFunctionType.SQRT).apply(UniFunctionType.MULINV).multiply(learningRate * miniBatchFactor)), M);
    }

}
