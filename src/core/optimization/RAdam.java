/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.optimization;

import utils.*;
import utils.matrix.DMatrix;
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
     * Learning rate for RAdam. Default value 0.001.
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
    private transient HashMap<Matrix, Integer> iters;

    /**
     * Maximum length of approximated SMA.
     *
     */
    private double pinf;

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
     * Default constructor for RAdam.
     *
     */
    public RAdam() {
        pinf = 2 / (1 - beta2) - 1;
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
     * Returns parameters used for Adam.
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
        iters = new HashMap<>();
        m = new HashMap<>();
        v = new HashMap<>();
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
        if (iters == null) iters = new HashMap<>();
        if (m == null) m = new HashMap<>();
        if (v == null) v = new HashMap<>();

        int iter;
        if (iters.containsKey(M)) iters.put(M, iter = iters.get(M) + 1);
        else iters.put(M, iter = 1);

        Matrix mM;
        if (m.containsKey(M)) mM = m.get(M);
        else m.put(M, mM = new DMatrix(M.getRows(), M.getCols()));

        Matrix vM;
        if (v.containsKey(M)) vM = v.get(M);
        else v.put(M, vM = new DMatrix(M.getRows(), M.getCols()));

        mM.multiply(beta1).add(dM.multiply(1 - beta1), mM);
        vM.multiply(beta2).add(dM.power(2).multiply(1 - beta2), vM);

        double beta1Iter = Math.pow(beta1, iter);
        double beta2Iter = Math.pow(beta2, iter);

        double stepSize = learningRate / (1 - beta1Iter);

        double pt = pinf - 2 * iter * beta2Iter / (1 - beta2Iter);
        if (pt > 4) {
            stepSize *= Math.sqrt(((pt - 4) * (pt - 2) * pinf) / ((pinf - 4) * (pinf - 2) * pt)) * Math.sqrt(1 - beta2Iter);
            M.subtract(mM.multiply(stepSize).divide(vM.apply(UnaryFunctionType.SQRT)), M);
        }
        else {
            M.subtract(mM.multiply(stepSize), M);
        }
    }

}

