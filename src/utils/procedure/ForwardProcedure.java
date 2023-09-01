/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure;

import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashSet;
import java.util.TreeMap;

/**
 * Defines interface for forward procedure.<br>
 *
 */
public interface ForwardProcedure {

    /**
     * Returns input matrix for procedure construction.
     *
     * @param resetPreviousInput if true resets previous input.
     * @return input matrices for procedure construction.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    TreeMap<Integer, Matrix> getInputMatrices(boolean resetPreviousInput) throws MatrixException;

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix getForwardProcedure() throws MatrixException, DynamicParamException;

    /**
     * Returns parameter matrices.
     *
     * @return parameter matrices.
     */
    HashSet<Matrix> getParameterMatrices();

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    HashSet<Matrix> getStopGradients();

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    HashSet<Matrix> getConstantMatrices();

    /**
     * Check if layer input is reversed.
     *
     * @return if true input layer input is reversed otherwise not.
     */
    boolean isReversedInput();

    /**
     * Returns true if input is joined otherwise returns false.
     *
     * @return true if input is joined otherwise returns false.
     */
    boolean isJoinedInput();

}
