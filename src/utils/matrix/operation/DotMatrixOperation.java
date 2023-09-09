/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements dot operation.
 *
 */
public class DotMatrixOperation extends AbstractMatrixOperation {

    /**
     * First matrix.
     *
     */
    private transient Matrix first;

    /**
     * Second matrix.
     *
     */
    private transient Matrix second;

    /**
     * Result matrix.
     *
     */
    private transient Matrix result;

    /**
     * Constructor for dot matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     */
    public DotMatrixOperation(int rows, int columns, int depth) {
        super(rows, columns, depth, false);
    }

    /**
     * Applies matrix operation.
     *
     * @param first first matrix.
     * @param second second matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if new mask dimensions or mask type are not matching with this mask.
     */
    public Matrix apply(Matrix first, Matrix second) throws MatrixException {
        this.first = first;
        this.second = second;
        this.result = first.getNewMatrix(first.getRows(), second.getColumns(), getDepth());
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
    protected Matrix getOther() {
        return second;
    }

    /**
     * Check if first matrix and optionally second matrix are masked at specific row and column.
     *
     * @param row row.
     * @param column column.
     * @param first first matrix.
     * @param second second matrix.
     * @return returns true if first or second matrix are masked at specific row and column.
     */
    protected boolean hasMaskAt(int row, int column, int depth, Matrix first, Matrix second) {
        return false;
    }

    /**
     * Applies matrix operation.
     *
     */
    protected void applyMatrixOperation() {
        final int rows1 = getRows();
        final Matrix other = getOther();
        final int rows2 = other.getRows();
        final int totalDepth = getDepth();
        final int rowStride = getStride();
        final int columnStride = getStride();
        final Matrix targetMatrix = getTargetMatrix();
        if (!hasMask(targetMatrix, other)) {
            for (int depth = 0; depth < totalDepth; depth++) {
                for (int row1 = 0; row1 < rows1; row1 += rowStride) {
                    for (int row2 = 0; row2 < rows2; row2 += rowStride) {
                        apply(row1, row2, depth, 0);
                    }
                }
            }
        }
        else {
            for (int depth = 0; depth < totalDepth; depth++) {
                for (int row1 = 0; row1 < rows1; row1 += rowStride) {
                    for (int row2 = 0; row2 < rows2; row2 += columnStride) {
                        if (!hasMaskAt(row1, row2, depth, targetMatrix, other)) {
                            applyMask(row1, row2, depth, 0);
                        }
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
     * @param depth current depth.
     * @param value current value.
     */
    public void apply(int row1, int row2, int depth, double value) {
        int columns = second.getColumns();
        for (int column = 0; column < columns; column++) {
            result.setValue(row1, column, depth, result.getValue(row1, column, depth) + first.getValue(row1, row2, depth) * second.getValue(row2, column, depth));
        }
    }

    /**
     * Applies operation assuming masked matrices.
     *
     * @param row1 current row1.
     * @param row2 current row2.
     * @param depth current depth.
     * @param value current value.
     */
    public void applyMask(int row1, int row2, int depth, double value) {
        int columns = second.getColumns();
        for (int column = 0; column < columns; column++) {
            if (!hasMaskAt(row1, row2, depth, first) && !hasMaskAt(row2, column, depth, second)) {
                result.setValue(row1, column, depth, result.getValue(row1, column, depth) + first.getValue(row1, row2, depth) * second.getValue(row2, column, depth));
            }
        }
    }

}
