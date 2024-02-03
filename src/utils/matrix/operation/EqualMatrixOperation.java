/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements equal matrix operation.
 *
 */
public class EqualMatrixOperation extends AbstractMatrixOperation {

    /**
     * Constructor for equal matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     */
    public EqualMatrixOperation(int rows, int columns, int depth) {
        super(rows, columns, depth, true);
    }

    /**
     * Applies matrix operation.
     *
     * @param first first matrix.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(Matrix first, Matrix result) throws MatrixException {
        if (result.getRows() != getRows() || result.getColumns() != getColumns() || result.getDepth() != getDepth()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns() + "x" + result.getDepth());
        }
        applyMatrixOperation(first, null, result);
    }

    /**
     * Applies operation.<br>
     * Ignores masking of other matrix.<br>
     *
     * @param row    current row.
     * @param column current column.
     * @param depth  current depth.
     * @param value  current value.
     * @param result result matrix.
     */
    public void apply(int row, int column, int depth, double value, Matrix result) {
        result.setValue(row, column, depth, value);
    }

}
