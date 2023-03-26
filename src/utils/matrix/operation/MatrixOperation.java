/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.MatrixException;

/**
 * Implements interface for matrix operation.
 *
 */
public interface MatrixOperation {

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     * @param value current value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void apply(int row, int column, int depth, double value) throws MatrixException;

    /**
     * Applies operation assuming masked matrices.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     * @param value current value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void applyMask(int row, int column, int depth, double value) throws MatrixException;

}
