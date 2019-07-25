/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.optimization;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.DMatrix;
import utils.Matrix;
import utils.MatrixException;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Class that implements Nesterov's Accelerated Gradient Descent optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/<br>
 *
 */
public class NesterovAcceleratedGradient implements Optimizer, Serializable {

    private static final long serialVersionUID = -783588127072068825L;

    /**
     * Learning rate for Nesterov Accelerated Gradient. Default value 0.001.
     *
     */
    private double learningRate = 0.001;

    /**
     * Momentum term for Nesterov Accelerated Gradient. Default value 0.0001.
     *
     */
    private double mu = 0.0001;

    /**
     * Hash map to store previous gradients.
     *
     */
    private transient HashMap<Matrix, Matrix> dPrev;

    /**
     * Hash map to store previous velocities.
     *
     */
    private transient HashMap<Matrix, Matrix> vPrev;

    /**
     * Relative size of mini batch.
     *
     */
    private double miniBatchFactor = 1;

    /**
     * Default constructor for Nesterov Accelerated Gradient.
     *
     */
    public NesterovAcceleratedGradient() {
    }

    /**
     * Constructor for Nesterov Accelerated Gradient.
     *
     * @param params parameters for Nesterov Accelerated Gradient.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public NesterovAcceleratedGradient(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Gets parameters used for Nesterov Accelerated Descent.
     *
     * @return parameters used for Nesterov Accelerated Descent.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("learningRate", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("mu", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for Nesterov Accelerated Descent.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - mu: mu (momentum) value for optimizer. Default value 0.0001.<br>
     *
     * @param params parameters used for Nesterov Accelerated Descent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("learningRate")) learningRate = params.getValueAsDouble("learningRate");
        if (params.hasParam("mu")) mu = params.getValueAsDouble("mu");
    }

    /**
     * Resets optimizer state.
     *
     */
    public void reset() {
        dPrev = new HashMap<>();
        vPrev = new HashMap<>();
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
        if (dPrev == null) dPrev = new HashMap<>();
        if (vPrev == null) vPrev = new HashMap<>();
        Matrix dMPrev;
        if (dPrev.containsKey(M)) dMPrev = dPrev.get(M);
        else dPrev.put(M, dMPrev = new DMatrix(M.getRows(), M.getCols()));

        Matrix vMPrev;
        if (vPrev.containsKey(M)) vMPrev = dPrev.get(M);
        else vPrev.put(M, vMPrev = new DMatrix(M.getRows(), M.getCols()));

        // vt=μvt−1−ϵ∇f(θt−1+μvt−1)
        Matrix vM = vMPrev.multiply(mu).subtract(dMPrev.add(vMPrev.multiply(mu)).multiply(learningRate * miniBatchFactor));

        M.add(vM, M);

        vPrev.put(M, vM);
        dPrev.put(M, dM);

    }

}

