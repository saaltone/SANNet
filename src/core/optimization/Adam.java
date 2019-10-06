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
 * Class that implements Adam optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/<br>
 *
 */
public class Adam implements Optimizer, Serializable {

    private static final long serialVersionUID = 2147864386790210492L;

    /**
     * Learning rate for Adam. Default value 0.001.
     *
     */
    private double learningRate = 0.001;

    /**
     * Beta1 term for Adam. Default value 0.9.
     *
     */
    private double beta1 = 0.9;

    /**
     * Beta2 term for Adam. Default value 0.999.
     *
     */
    private double beta2 = 0.999;

    /**
     * Optimizer iteration count for Adam.
     *
     */
    private transient int iter = 1;

    /**
     * Hash map to store first moments (means).
     *
     */
    private transient HashMap<Matrix, Matrix> m;

    /**
     * Hash map to store second moments (uncentered variances).
     *
     */
    private transient HashMap<Matrix, Matrix> v;

    /**
     * Relative size of mini batch.
     *
     */
    private double miniBatchFactor = 1;

    /**
     * Default constructor for Adam.
     *
     */
    public Adam() {
    }

    /**
     * Constructor for Adam.
     *
     * @param params parameters for Adam.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Adam(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Gets parameters used for Adam.
     *
     * @return parameters used for Adam.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("learningRate", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("beta1", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("beta2", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for Adam.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - beta1: beta1 value for optimizer. Default value 0.9.<br>
     *     - beta2: beta2 value for optimizer. Default value 0.999.<br>
     *
     * @param params parameters used for Adam.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("learningRate")) learningRate = params.getValueAsDouble("learningRate");
        if (params.hasParam("beta1")) beta1 = params.getValueAsDouble("beta1");
        if (params.hasParam("beta2")) beta2 = params.getValueAsDouble("beta2");
    }

    /**
     * Resets optimizer state.
     *
     */
    public void reset() {
        m = new HashMap<>();
        v = new HashMap<>();
        iter = 1;
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
     * Set iteration count.
     *
     * @param iter iteration count.
     */
    public void setIteration(int iter) {
        this.iter = iter;
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
        if (m == null) m = new HashMap<>();
        if (v == null) v = new HashMap<>();
        if (iter == 0) iter = 1;
        Matrix mM;
        if (m.containsKey(M)) mM = m.get(M);
        else m.put(M, mM = new DMatrix(M.getRows(), M.getCols()));

        Matrix vM;
        if (v.containsKey(M)) vM = v.get(M);
        else v.put(M, vM = new DMatrix(M.getRows(), M.getCols()));

        // mt = β1*mt − 1 + (1 − β1)*gt
        mM.multiply(beta1).add(dM.multiply(1 - beta1), mM);

        // vt = β2*vt − 1 + (1 − β2)*g2t
        vM.multiply(beta2).add(dM.power(2).multiply(1 - beta2), vM);

        // mt = mt / (1 − βt1)
        Matrix mM_hat = mM.divide(1 - Math.pow(beta1, iter));

        // vt = vt / (1 − βt2)
        Matrix vM_hat = vM.divide(1 - Math.pow(beta2, iter));

        // θt+1 = θt − η / (√^vt + ϵ) * mt
        /**
         * Epsilon term for Adam. Default value 10E-8.<br>
         * Term provides mathematical stability for optimizer.<br>
         *
         */
        double epsilon = 10E-8;
        M.subtract(vM_hat.add(epsilon).apply(UniFunctionType.SQRT).apply(UniFunctionType.MULINV).multiply(mM_hat).multiply(learningRate * miniBatchFactor), M);

        iter++;
    }

}

