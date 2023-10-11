/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements sum matrix operation.
 *
 */
public class SumMatrixOperation extends AbstractMatrixOperation {

    /**
     * Cumulated value.
     *
     */
    private transient double value;

    /**
     * Number of counted entries.
     *
     */
    private transient int count;

    /**
     * Constructor for sum matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     */
    public SumMatrixOperation(int rows, int columns, int depth) {
        super(rows, columns, depth, true);
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
        this.value += value;
        count++;
    }

    /**
     * Applies sum operation.
     *
     * @param first first matrix.
     * @return sum of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double applySum(Matrix first) throws MatrixException {
        value = 0;
        count = 0;
        applyMatrixOperation(first, null, null);
        return value;
    }

    /**
     * Applies mean operation.
     *
     * @param first first matrix.
     * @return mean of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double applyMean(Matrix first) throws MatrixException {
        applyMatrixOperation(first, null, null);
        return value / (double)count;
    }

}
