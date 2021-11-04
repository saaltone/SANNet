/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.procedure;

import utils.configurable.DynamicParamException;
import utils.matrix.MMatrix;
import utils.matrix.MatrixException;

/**
 * Defines interface for forward procedure.<br>
 *
 */
public interface ForwardProcedure {

    /**
     * Returns input matrix for procedure construction.
     *
     * @param resetPreviousInput if true resets previous input.
     * @return input matrix for procedure construction.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    MMatrix getInputMatrices(boolean resetPreviousInput) throws MatrixException;

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    MMatrix getForwardProcedure() throws MatrixException, DynamicParamException;

}
