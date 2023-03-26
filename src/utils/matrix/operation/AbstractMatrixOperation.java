/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;

/**
 * Implements abstract matrix operation used by all matrix operations.
 *
 */
public abstract class AbstractMatrixOperation implements MatrixOperation, Serializable {

    @Serial
    private static final long serialVersionUID = 4515327729821343316L;

    /**
     * Number of rows for operation.
     *
     */
    private final int rows;

    /**
     * Number of columns for operation.
     *
     */
    private final int columns;

    /**
     * Number of columns for operation.
     *
     */
    private final int depth;

    /**
     * If true operation provides value when applying operation otherwise false.
     *
     */
    private final boolean provideValue;

    /**
     * Stride step for operation.
     *
     */
    private final int stride;

    /**
     * Constructor for abstract matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param provideValue if true operation provides value when applying operation otherwise false.
     */
    public AbstractMatrixOperation(int rows, int columns, int depth, boolean provideValue) {
        this.rows = rows;
        this.columns = columns;
        this.depth = depth;
        this.provideValue = provideValue;
        this.stride = 1;
    }

    /**
     * Constructor for abstract matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param provideValue if true operation provides value when applying operation otherwise false.
     * @param stride stride step for operation.
     */
    public AbstractMatrixOperation(int rows, int columns, int depth, boolean provideValue, int stride) {
        this.rows = rows;
        this.columns = columns;
        this.depth = depth;
        this.provideValue = provideValue;
        this.stride = stride;
    }

    /**
     * Returns number of rows for operation.
     *
     * @return number of rows for operation.
     */
    protected int getRows() {
        return rows;
    }

    /**
     * Returns number of columns for operation.
     *
     * @return number of columns for operation.
     */
    protected int getColumns() {
        return columns;
    }

    /**
     * Returns number of columns for operation.
     *
     * @return number of columns for operation.
     */
    protected int getDepth() {
        return depth;
    }

    /**
     * If true operation provides value when applying operation otherwise false.
     *
     * @return if true operation provides value when applying operation otherwise false.
     */
    protected boolean getProvideValue() {
        return provideValue;
    }

    /**
     * Returns other matrix used in operation.
     *
     * @return other matrix used in operation.
     */
    protected abstract Matrix getOther();

    /**
     * Returns stride step for operation.
     *
     * @return stride step for operation.
     */
    protected int getStride() {
        return stride;
    }

    /**
     * Applies operation assuming masked matrices.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     * @param value current value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyMask(int row, int column, int depth, double value) throws MatrixException {
        apply(row, column, depth, value);
    }

    /**
     * Applies matrix operation.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void applyMatrixOperation() throws MatrixException {
        final int rows = getRows();
        final int columns = getColumns();
        final int totalDepth = getDepth();
        final Matrix other = getOther();
        final boolean provideValue = getProvideValue();
        final int rowStride = getStride();
        final int columnStride = getStride();
        final Matrix targetMatrix = getTargetMatrix();
        if (!hasMask(targetMatrix, other)) {
            if (provideValue) {
                for (int depth = 0; depth < totalDepth; depth++) {
                    for (int row = 0; row < rows; row += rowStride) {
                        for (int column = 0; column < columns; column += columnStride) {
                            apply(row, column, depth, targetMatrix.getValue(row, column, depth));
                        }
                    }
                }
            }
            else {
                for (int depth = 0; depth < totalDepth; depth++) {
                    for (int row = 0; row < rows; row += rowStride) {
                        for (int column = 0; column < columns; column += columnStride) {
                            apply(row, column, depth, 0);
                        }
                    }
                }
            }
        }
        else {
            if (provideValue) {
                for (int depth = 0; depth < totalDepth; depth++) {
                    for (int row = 0; row < rows; row += rowStride) {
                        for (int column = 0; column < columns; column += columnStride) {
                            if (!hasMaskAt(row, column, depth, targetMatrix, other)) {
                                applyMask(row, column, depth, targetMatrix.getValue(row, column, depth));
                            }
                        }
                    }
                }
            }
            else {
                for (int depth = 0; depth < totalDepth; depth++) {
                    for (int row = 0; row < rows; row += rowStride) {
                        for (int column = 0; column < columns; column += columnStride) {
                            if (!hasMaskAt(row, column, depth, targetMatrix, other)) {
                                applyMask(row, column, depth, 0);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Returns target matrix.
     *
     * @return target matrix.
     */
    protected abstract Matrix getTargetMatrix();

    /**
     * Checks if first matrix and optionally second matrix are masked.
     *
     * @param first first matrix.
     * @param second second matrix.
     * @return returns true if first or second matrix are masked.
     */
    protected boolean hasMask(Matrix first, Matrix second) {
        return first.getMask() != null || (second != null && second.getMask() != null);
    }

    /**
     * Check if first matrix and optionally second matrix are masked at specific row and column.
     *
     * @param row row.
     * @param column column.
     * @param depth depth.
     * @param first first matrix.
     * @param second second matrix.
     * @return returns true if first or second matrix are masked at specific row and column.
     */
    protected boolean hasMaskAt(int row, int column, int depth, Matrix first, Matrix second) {
        return first.hasMaskAt(row, column, depth) || (second != null && second.hasMaskAt(row, column, depth));
    }

    /**
     * Check if matrix is masked at specific row and column.
     *
     * @param row row.
     * @param column column.
     * @param depth depth.
     * @param matrix matrix.
     * @return returns true if first or second matrix is masked at specific row and column.
     */
    protected boolean hasMaskAt(int row, int column, int depth, Matrix matrix) {
        return matrix.hasMaskAt(row, column, depth);
    }

}
