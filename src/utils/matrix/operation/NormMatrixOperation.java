/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements norm matrix operation.
 *
 */
public class NormMatrixOperation extends AbstractMatrixOperation {

    /**
     * Power for norm operation.
     *
     */
    private final int p;

    /**
     * Cumulated norm value.
     *
     */
    private double value;

    /**
     * Constructor for norm matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param p power for norm operation.
     */
    public NormMatrixOperation(int rows, int columns, int depth, int p) {
        super(rows, columns, depth, true);
        this.p = p;
    }

    /**
     * Applies operation.
     *
     * @param first first matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double apply(Matrix first) throws MatrixException {
        value = 0;
        applyMatrixOperation(first, null, null);
        return Math.pow(value, 1 / (double)p);
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
        this.value += Math.pow(Math.abs(value), p);
    }

}
