/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Defines Softmax gradient matrix operation.
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
     */
    public SoftmaxGradientMatrixOperation(int rows, int columns) {
        super(rows, columns, false);
    }

    /**
     * Applies operation.
     *
     * @param first first matrix.
     * @param result result matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix first, Matrix result) throws MatrixException {
        this.first = first;
        this.result = result;
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
    public Matrix getAnother() {
        return null;
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param row1 current row1.
     * @param value current value.
     */
    public void apply(int row, int row1, double value) {
        result.setValue(row1, row, (row == row1 ? 1 : 0) - first.getValue(row1, 0));
    }

}
