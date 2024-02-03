/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements normalize matrix operation.
 *
 */
public class NormalizeMatrixOperation extends AbstractMatrixOperation {

    /**
     * Mean for normalize operation.
     *
     */
    private final double mean;

    /**
     * Variance for normalize operation.
     *
     */
    private final double variance;

    /**
     * Constructor for normalize matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param mean mean for normalize operation.
     * @param variance variance for normalize operation.
     */
    public NormalizeMatrixOperation(int rows, int columns, int depth, double mean, double variance) {
        super(rows, columns, depth, true);
        this.mean = mean;
        this.variance = variance;
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
        return applyMatrixOperation(first, null, result);
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
        result.setValue(row, column, depth, (value - mean) / variance);
    }

}
