/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Defines entropy matrix operation.
 *
 */
public class EntropyMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input matrix.
     *
     */
    private Matrix input;

    /**
     * Cumulated value.
     *
     */
    private double value;

    /**
     * Number of counted entries.
     *
     */
    private int count;

    /**
     * Constructor for entropy matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     */
    public EntropyMatrixOperation(int rows, int columns) {
        super(rows, columns, true);
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
        this.value += value * Math.log10(value) / Math.log10(2);
        count++;
    }

    /**
     * Applies entropy operation.
     *
     * @param input input matrix.
     * @return entropy of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double applyEntropy(Matrix input) throws MatrixException {
        this.input = input;
        applyMatrixOperation();
        return -value / (double)count;
    }

    /**
     * Returns target matrix.
     *
     * @return target matrix.
     */
    protected Matrix getTargetMatrix() {
        return input;
    }

}
