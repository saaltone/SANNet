/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.optimization;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Matrix;
import utils.MatrixException;

/**
 * Defines interface for optimizers.
 *
 */
public interface Optimizer {

    /**
     * Sets parameters for optimizer.
     *
     * @param params parameters for optimizer
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void setParams(DynamicParam params) throws DynamicParamException;

    /**
     * Resets optimizer state.
     *
     */
    void reset();

    /**
     * Sets relative size of mini batch.
     *
     * @param miniBatchFactor relative size of mini batch.
     */
    void setMiniBatchFactor(double miniBatchFactor);

    /**
     * Optimizes given weight (W) and bias (B) pair with given gradients respectively.
     *
     * @param W weight matrix to be optimized.
     * @param dW weight gradients for optimization step.
     * @param B bias matrix to be optimized.
     * @param dB bias gradients for optimization step.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void optimize(Matrix W, Matrix dW, Matrix B, Matrix dB) throws MatrixException;

    /**
     * Optimizes single matrix (M) using calculated matrix gradient (dM).<br>
     * Matrix can be for example weight or bias matrix with gradient.<br>
     *
     * @param M matrix to be optimized.
     * @param dM matrix gradients for optimization step.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void optimize(Matrix M, Matrix dM) throws MatrixException;

}

