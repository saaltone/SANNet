/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.optimization;

import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Interface for optimizers.<br>
 *
 */
public interface Optimizer {

    /**
     * Returns parameters of optimizer.
     *
     * @return parameters for optimizer.
     */
    String getParams();

    /**
     * Resets optimizer state.
     *
     */
    void reset();

    /**
     * Optimizes single matrix (M) using calculated matrix gradient (dM).<br>
     * Matrix can be for example weight or bias matrix with gradient.<br>
     *
     * @param matrix matrix to be optimized.
     * @param matrixGradient matrix gradients for optimization step.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void optimize(Matrix matrix, Matrix matrixGradient) throws MatrixException, DynamicParamException;

    /**
     * Returns name of optimizer.
     *
     * @return name of optimizer.
     */
    String getName();

}

