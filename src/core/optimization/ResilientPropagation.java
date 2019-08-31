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
 * Class that implements Resilient Propagation optimizer.<br>
 * Resilient Propagation is full batch algorithm.<br>
 * <br>
 * Reference: http://130.243.105.49/~lilien/ml/seminars/2007_03_12c-Markus_Ingvarsson-RPROP.pdf<br>
 *
 */
public class ResilientPropagation implements Optimizer, Serializable {

    private static final long serialVersionUID = -5041801098584596493L;

    /**
     * Hash map to store previous gradients.
     *
     */
    private transient HashMap<Matrix, Matrix> dPrev;

    /**
     * Hash map to store previous directions.
     *
     */
    private transient HashMap<Matrix, Matrix> wPrev;

    /**
     * Default constructor for Resilient Propagation.
     *
     */
    public ResilientPropagation() {
    }

    /**
     * Constructor for Resilient Propagation.
     *
     * @param params parameters for Resilient Propagation (not relevant).
     */
    public ResilientPropagation(String params) {
        setParams(null);
    }

    /**
     * Sets parameters used for Resilient Propagation. Not needed for Resilient Propagation.
     *
     * @param params parameters used for Resilient Propagation (not relevant).
     */
    public void setParams(DynamicParam params) {
    }

    /**
     * Resets optimizer state.
     *
     */
    public void reset() {
        dPrev = new HashMap<>();
        wPrev = new HashMap<>();
    }

    /**
     * Sets relative size of mini batch.
     *
     * @param miniBatchFactor relative size of mini batch.
     */
    public void setMiniBatchFactor(double miniBatchFactor) {

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
        if (wPrev == null) wPrev = new HashMap<>();
        Matrix dMPrev;
        if (dPrev.containsKey(M)) dMPrev = dPrev.get(M);
        else dPrev.put(M, dMPrev = dM);

        Matrix WPrev;
        if (wPrev.containsKey(M)) WPrev = wPrev.get(M);
        else wPrev.put(M, WPrev = new DMatrix(M.getRows(), M.getCols()));

        Matrix dWDir = dMPrev.sgnmul(dM);

        Matrix.MatrixBiOperation rpropRule = (value1, value2) -> value1 == -1 ? Math.max(0.5 * value2, 10E-6) : value1 == 1 ? Math.min(1.2 * value2, 50) : value2;
        wPrev.put(M, WPrev = dWDir.applyBi(WPrev, rpropRule));

        M.subtract(dM.sgn().multiply(WPrev), M);

        dPrev.put(M, dWDir.applyBi(dM, (value1, value2) -> value1 == -1 ? 0 : value2));
    }

}
