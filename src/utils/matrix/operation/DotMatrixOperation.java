/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;

/**
 * Implements dot operation.
 *
 */
public class DotMatrixOperation extends AbstractMatrixOperation {

    /**
     * First matrix.
     *
     */
    private Matrix first;

    /**
     * Second matrix.
     *
     */
    private Matrix second;

    /**
     * Result matrix.
     *
     */
    private Matrix result;

    /**
     * Constructor for dot matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     */
    public DotMatrixOperation(int rows, int columns) {
        super(rows, columns, false);
    }

    /**
     * Applies matrix operation.
     *
     * @param first first matrix.
     * @param second second matrix.
     * @param result result matrix.
     * @return result matrix.
     */
    public Matrix apply(Matrix first, Matrix second, Matrix result) {
        this.first = first;
        this.second = second;
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
    protected Matrix getAnother() {
        return second;
    }

    /**
     * Check if first matrix and optionally second matrix has mask at specific row and column.
     *
     * @param row row.
     * @param column column.
     * @param first first matrix.
     * @param second second matrix.
     * @return returns true if first or second matrix has mask at specific row and column.
     */
    protected boolean hasMaskAt(int row, int column, Matrix first, Matrix second) {
        return false;
    }

    /**
     * Applies matrix operation.
     *
     */

    protected void applyMatrixOperation() {
        final int rows1 = getRows();
        final Matrix other = getAnother();
        final int rows2 = other.getRows();
        final int rowStride = getStride();
        final int columnStride = getStride();
        final Matrix targetMatrix = getTargetMatrix();
        if (!hasMask(targetMatrix, other)) {
            for (int row1 = 0; row1 < rows1; row1 += rowStride) {
                for (int row2 = 0; row2 < rows2; row2 += rowStride) {
                    apply(row1, row2, 0);
                }
            }
        }
        else {
            for (int row1 = 0; row1 < rows1; row1 += rowStride) {
                for (int row2 = 0; row2 < rows2; row2 += columnStride) {
                    if (!hasMaskAt(row1, row2, targetMatrix, other)) {
                        applyMask(row1, row2, 0);
                    }
                }
            }
        }
    }

    /**
     * Applies operation.
     *
     * @param row1 current row1.
     * @param row2 current row2.
     * @param value current value.
     */
    public void apply(int row1, int row2, double value) {
        int cols = second.getColumns();
        for (int col = 0; col < cols; col++) {
            result.setValue(row1, col, result.getValue(row1, col) + first.getValue(row1, row2) * second.getValue(row2, col));
        }
    }

    /**
     * Applies operation assuming masked matrices.
     *
     * @param row1 current row1.
     * @param row2 current row2.
     * @param value current value.
     */
    public void applyMask(int row1, int row2, double value) {
        int cols = second.getColumns();
        for (int col = 0; col < cols; col++) {
            if (!hasMaskAt(row1, row2, first) && !hasMaskAt(row2, col, second)) {
                result.setValue(row1, col, result.getValue(row1, col) + first.getValue(row1, row2) * second.getValue(row2, col));
            }
        }
    }

}
