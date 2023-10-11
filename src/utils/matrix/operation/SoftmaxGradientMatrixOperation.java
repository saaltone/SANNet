/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements Softmax gradient matrix operation.
 *
 */
public class SoftmaxGradientMatrixOperation extends AbstractMatrixOperation {

    /**
     * First matrix.
     *
     */
    private transient Matrix first;

    /**
     * Constructor for Softmax gradient matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     */
    public SoftmaxGradientMatrixOperation(int rows, int columns, int depth) {
        super(rows, columns, depth, false);
    }

    /**
     * Applies operation.
     *
     * @param first first matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix first) throws MatrixException {
        this.first = first;
        return applyMatrixOperation(first, null, first.getNewMatrix(first.getRows(), first.getRows(), getDepth()));
    }

    /**
     * Applies operation.
     *
     * @param row    current row.
     * @param row1   current row1.
     * @param depth  current depth.
     * @param value  current value.
     * @param result result matrix.
     */
    public void apply(int row, int row1, int depth, double value, Matrix result) {
        result.setValue(row1, row, depth, (row == row1 ? 1 : 0) - first.getValue(row1, 0, depth));
    }

}
