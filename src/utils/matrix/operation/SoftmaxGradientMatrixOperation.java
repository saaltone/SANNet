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
    private Matrix first;

    /**
     * Result matrix.
     *
     */
    private Matrix result;

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
        this.result = first.getNewMatrix(first.getRows(), first.getRows(), getDepth());
        applyMatrixOperation();
        return result;
    }

    /**
     * Returns target matrix.
     *
     * @return target matrix.
     */
    protected Matrix getTargetMatrix() {
        return first;
    }

    /**
     * Returns another matrix used in operation.
     *
     * @return another matrix used in operation.
     */
    public Matrix getOther() {
        return null;
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param row1 current row1.
     * @param depth current depth.
     * @param value current value.
     */
    public void apply(int row, int row1, int depth, double value) {
        result.setValue(row1, row, depth, (row == row1 ? 1 : 0) - first.getValue(row1, 0, depth));
    }

}
