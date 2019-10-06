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
 * Class that implements RMSProp optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/<br>
 *
 */
public class RMSProp implements Optimizer, Serializable {

    private static final long serialVersionUID = 3251200097077919746L;

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
     * Relative size of mini batch.
     *
     */
    private double miniBatchFactor = 1;

    /**
     * Default constructor for RMSProp.
     *
     */
    public RMSProp() {
    }

    /**
     * Constructor for RMSProp.
     *
     * @param params parameters for RMSProp.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public RMSProp(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Gets parameters used for RMSProp.
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
        if (eg2 == null) eg2 = new HashMap<>();
        Matrix mEg2;
        if (eg2.containsKey(M)) mEg2 = eg2.get(M);
        else eg2.put(M, mEg2 = new DMatrix(M.getRows(), M.getCols()));

        eg2.put(M, mEg2 = mEg2.multiply(gamma).add(dM.power(2).multiply(1 - gamma)));

        /**
         * Epsilon term for RMSProp. Default value 10E-8.<br>
         * Term provides mathematical stability for optimizer.<br>
         *
         */
        double epsilon = 10E-8;
        M.subtract(mEg2.add(epsilon).apply(UniFunctionType.SQRT).apply(UniFunctionType.MULINV).multiply(learningRate * miniBatchFactor).multiply(dM), M);
    }

}

