/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Defines dot operation.
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
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix first, Matrix second, Matrix result) throws MatrixException {
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
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     */
    public void apply(int row, int column, double value) {
        int xSize = first.getColumns();
        for (int x = 0; x < xSize; x++) {
            result.setValue(row, column, result.getValue(row, column) + first.getValue(row, x) * second.getValue(x, column));
        }
    }

    /**
     * Applies operation assuming masked matrices.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     */
    public void applyMask(int row, int column, double value) {
        int xSize = first.getColumns();
        double sumValue = result.getValue(row, column);
        for (int x = 0; x < xSize; x++) {
            if (!hasMaskAt(row, x, first) && !hasMaskAt(x, column, second)) {
                sumValue += first.getValue(row, x) * second.getValue(x, column);
            }
        }
        result.setValue(row, column, sumValue);
    }

}
