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
import utils.matrix.UnaryFunctionType;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Class that implements Adamax optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/<br>
 *
 */
public class Adamax implements Optimizer, Serializable {

    private static final long serialVersionUID = 9136132997261066936L;

    /**
     * Learning rate for Adamax. Default value 0.001.
     *
     */
    private double learningRate = 0.001;

    /**
     * Epsilon term for Adamax. Default value 10E-8.<br>
     * Term provides mathematical stability for optimizer.<br>
     *
     */
    private final double epsilon = 10E-8;

    /**
     * Beta1 term for Adamax. Default value 0.9.
     *
     */
    private double beta1 = 0.9;

    /**
     * Beta2 term for Adamax. Default value 0.999.
     *
     */
    private double beta2 = 0.999;

    /**
     * Optimizer iteration count for Adamax.
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
     * Default constructor for Adamax.
     *
     */
    public Adamax() {
    }

    /**
     * Constructor for Adamax.
     *
     * @param params parameters for Adamax.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Adamax(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for Adamax.
     *
     * @return parameters used for Adamax.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("learningRate", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("beta1", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("beta2", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for Adamax.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - beta1: beta1 value for optimizer. Default value 0.9.<br>
     *     - beta2: beta2 value for optimizer. Default value 0.999.<br>
     *
     * @param params parameters used for Adamax.
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

        Matrix dM_abs = dM.apply(UnaryFunctionType.ABS);

        // mt = β1*mt − 1 + (1 − β1)*gt
        mM.multiply(beta1).add(dM.multiply(1 - beta1), mM);

        // vt = β2*vt − 1 + (1 − β2)*|gt|
        vM.multiply(beta2).add(dM_abs.multiply(1 - beta2), vM);

        // mt = mt / (1 − βt1)
        Matrix mM_hat = mM.divide(1 - Math.pow(beta1, iter));

        // ut= max(β2⋅vt−1,|gt|)
        Matrix uM = vM.multiply(beta2).max(dM_abs);

        // θt+1 = θt − η / ut * mt
        M.subtract(uM.apply(UnaryFunctionType.MULINV).multiply(mM_hat).multiply(learningRate * miniBatchFactor), M);

        iter++;
    }

}

