/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;

/**
 * Defines abstract matrix operation used by all matrix operations.
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
     * @param provideValue if true operation provides value when applying operation otherwise false.
     */
    public AbstractMatrixOperation(int rows, int columns, boolean provideValue) {
        this.rows = rows;
        this.columns = columns;
        this.provideValue = provideValue;
        this.stride = 1;
    }

    /**
     * Constructor for abstract matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param provideValue if true operation provides value when applying operation otherwise false.
     * @param stride stride step for operation.
     */
    public AbstractMatrixOperation(int rows, int columns, boolean provideValue, int stride) {
        this.rows = rows;
        this.columns = columns;
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
     * If true operation provides value when applying operation otherwise false.
     *
     * @return if true operation provides value when applying operation otherwise false.
     */
    protected boolean getProvideValue() {
        return provideValue;
    }

    /**
     * Returns another matrix used in operation.
     *
     * @return another matrix used in operation.
     */
    protected abstract Matrix getAnother();

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
     * @param value current value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyMask(int row, int column, double value) throws MatrixException {
        apply(row, column, value);
    }

    /**
     * Applies matrix operation.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void applyMatrixOperation() throws MatrixException {
        final int rows = getRows();
        final int columns = getColumns();
        final Matrix other = getAnother();
        final boolean provideValue = getProvideValue();
        final int rowStride = getStride();
        final int columnStride = getStride();
        final Matrix targetMatrix = getTargetMatrix();
        if (!hasMask(targetMatrix, other)) {
            if (provideValue) {
                for (int row = 0; row < rows; row += rowStride) {
                    for (int column = 0; column < columns; column += columnStride) {
                        apply(row, column, targetMatrix.getValue(row, column));
                    }
                }
            }
            else {
                for (int row = 0; row < rows; row += rowStride) {
                    for (int column = 0; column < columns; column += columnStride) {
                        apply(row, column, 0);
                    }
                }
            }
        }
        else {
            if (provideValue) {
                for (int row = 0; row < rows; row += rowStride) {
                    for (int column = 0; column < columns; column += columnStride) {
                        if (!hasMaskAt(row, column, targetMatrix, other)) {
                            apply(row, column, targetMatrix.getValue(row, column));
                        }
                    }
                }
            }
            else {
                for (int row = 0; row < rows; row += rowStride) {
                    for (int column = 0; column < columns; column += columnStride) {
                        if (!hasMaskAt(row, column, targetMatrix, other)) {
                            apply(row, column, 0);
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
     * Check if first matrix and optionally second matrix has mask.
     *
     * @param first first matrix.
     * @param second second matrix.
     * @return returns true if first or second matrix has mask.
     */
    protected boolean hasMask(Matrix first, Matrix second) {
        return first.getMask() != null || (second != null && second.getMask() != null);
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
        return first.hasMaskAt(row, column) || (second != null && second.hasMaskAt(row, column));
    }

    /**
     * Check if matrix has mask at specific row and column.
     *
     * @param row row.
     * @param column column.
     * @param matrix matrix.
     * @return returns true if first or second matrix has mask at specific row and column.
     */
    protected boolean hasMaskAt(int row, int column, Matrix matrix) {
        return matrix.hasMaskAt(row, column);
    }

}
