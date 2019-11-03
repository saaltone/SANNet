/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.optimization;

import utils.*;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Class that implements Gradient Descent with Momentum optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/<br>
 *
 */
public class MomentumGradientDescent implements Optimizer, Serializable {

    private static final long serialVersionUID = -983868918422365256L;

    /**
     * Learning rate for Momentum Gradient Descent. Default value 0.001.
     *
     */
    private double learningRate = 0.001;

    /**
     * Momentum term for Momentum Gradient Descent. Default value 0.0001.
     *
     */
    private double mu = 0.0001;

    /**
     * Hash map to store previous gradients.
     *
     */
    private transient HashMap<Matrix, Matrix> dPrev;

    /**
     * Relative size of mini batch.
     *
     */
    private double miniBatchFactor = 1;

    /**
     * Default constructor for Momentum Gradient Descent.
     *
     */
    public MomentumGradientDescent() {
    }

    /**
     * Constructor for Momentum Gradient Descent.
     *
     * @param params parameters for Momentum Gradient Descent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MomentumGradientDescent(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for Momentum Gradient Descent.
     *
     * @return parameters used for Momentum Gradient Descent.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("learningRate", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("mu", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for Momentum Gradient Descent.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - mu: mu (momentum) value for optimizer. Default value 0.0001.<br>
     *
     * @param params parameters used for Momentum Gradient Descent.
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
        Matrix dMPrev;
        if (dPrev.containsKey(M)) dMPrev = dPrev.get(M);
        else dPrev.put(M, dMPrev = new DMatrix(M.getRows(), M.getCols()));

        // θt+1=θt+μtvt−εt∇f(θt)
        Matrix dMDelta = dMPrev.multiply(mu).subtract(dM.multiply(learningRate * miniBatchFactor));

        M.add(dMDelta, M);

        // vt+1=μtvt−εt∇f(θt)
        dPrev.put(M, dMDelta);

    }

}

