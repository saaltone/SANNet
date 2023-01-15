/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements matrix unjoin operation.
 *
 */
public class UnjoinMatrixOperation extends AbstractMatrixOperation {

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
     * Unjoins at defined row.
     *
     */
    private final int unjoinAtRow;

    /**
     * Unjoins at defined column.
     *
     */
    private final int unjoinAtColumn;

    /**
     * Constructor for join binary operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param unjoinAtRow unjoins at row.
     * @param unjoinAtColumn unjoins at column.
     */
    public UnjoinMatrixOperation(int rows, int columns, int unjoinAtRow, int unjoinAtColumn) {
        super(rows, columns, true);
        this.unjoinAtRow = unjoinAtRow;
        this.unjoinAtColumn = unjoinAtColumn;
    }

    /**
     * Applies matrix operation.
     *
     * @param first first matrix.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(Matrix first, Matrix result) throws MatrixException {
        this.first = first;
        this.result = result;
        applyMatrixOperation();
    }

    /**
     * Calculates gradient.
     *
     * @param first first matrix.
     * @param outputGradient output gradient.
     * @return input gradient
     */
    public Matrix applyGradient(Matrix first, Matrix outputGradient) {
        Matrix result = new DMatrix(first.getRows(), first.getColumns());
        final int rows = getRows();
        final int columns = getColumns();
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                result.setValue(unjoinAtRow + row, unjoinAtColumn + column, outputGradient.getValue(row, column));
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
        result.setValue(row, column, first.getValue(unjoinAtRow + row, unjoinAtColumn + column));
    }

}
