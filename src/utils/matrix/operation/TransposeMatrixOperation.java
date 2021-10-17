/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Mask;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements matrix transpose operation.
 *
 */
public class TransposeMatrixOperation extends AbstractMatrixOperation {

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
     * Constructor for matrix unary operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     */
    public TransposeMatrixOperation(int rows, int columns) {
        super(rows, columns, true);
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
        if (first.getMask() != null) {
            result.setMask();
            Mask resultMask = result.getMask();
            int rows = first.getRows();
            int columns = first.getColumns();
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    if (first.hasMaskAt(row, column)) resultMask.setMask(column, row, true);
                }
            }
        }
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
     * @param column current column.
     * @param value current value.
     */
    public void apply(int row, int column, double value) {
        result.setValue(column, row, value);
    }

}
