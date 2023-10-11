/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements flatten matrix operation.
 *
 */
public class UnflattenMatrixOperation extends AbstractMatrixOperation {


    /**
     * Constructor for unflatten matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     */
    public UnflattenMatrixOperation(int rows, int columns, int depth) {
        super(rows, columns, depth, false);
    }


    /**
     * Applies matrix operation.
     *
     * @param first first matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix first) throws MatrixException {
        return first.redimension(getRows(), getColumns(), getDepth(), false);
    }

    /**
     * Applies operation.
     *
     * @param row    current row.
     * @param column current column.
     * @param depth  current depth.
     * @param value  current value.
     * @param result result matrix.
     */
    public void apply(int row, int column, int depth, double value, Matrix result) {
    }

}
