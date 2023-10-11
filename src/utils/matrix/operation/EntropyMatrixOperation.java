/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements entropy matrix operation.
 *
 */
public class EntropyMatrixOperation extends AbstractMatrixOperation {

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
     * Constructor for entropy matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     */
    public EntropyMatrixOperation(int rows, int columns, int depth) {
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
        this.value += value * Math.log10(value) / Math.log10(2);
        count++;
    }

    /**
     * Applies entropy operation.
     *
     * @param first first matrix.
     * @return entropy of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double applyEntropy(Matrix first) throws MatrixException {
        value = 0;
        count = 0;
        applyMatrixOperation(first, null, null);
        return -value / (double)count;
    }

}
