/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
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
    private transient Matrix first;

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
     * Unjoins at defined depth.
     *
     */
    private final int unjoinAtDepth;

    /**
     * Constructor for join binary operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param unjoinAtRow unjoins at row.
     * @param unjoinAtColumn unjoins at column.
     * @param unjoinAtDepth unjoins at depth.
     */
    public UnjoinMatrixOperation(int rows, int columns, int depth, int unjoinAtRow, int unjoinAtColumn, int unjoinAtDepth) {
        super(rows, columns, depth, true);
        this.unjoinAtRow = unjoinAtRow;
        this.unjoinAtColumn = unjoinAtColumn;
        this.unjoinAtDepth = unjoinAtDepth;
    }

    /**
     * Applies matrix operation.
     *
     * @param first first matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix first) throws MatrixException {
        this.first = first;
        return applyMatrixOperation(first, null, first.getNewMatrix(getRows(), getColumns(), getDepth()));
    }

    /**
     * Calculates gradient.
     *
     * @param outputGradient output gradient.
     * @return input gradient
     */
    public Matrix applyGradient(Matrix outputGradient) {
        final int rows = getRows();
        final int columns = getColumns();
        final int totalDepth = getDepth();
        Matrix result = new DMatrix(rows, columns, totalDepth);
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                for (int depth = 0; depth < totalDepth; depth++) {
                    result.setValue(unjoinAtRow + row, unjoinAtColumn + column, unjoinAtDepth + depth, outputGradient.getValue(row, column, depth));
                }
            }
        }
        return result;
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
        result.setValue(row, column, depth, first.getValue(unjoinAtRow + row, unjoinAtColumn + column, unjoinAtDepth + depth));
    }

}
