/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Defines sum matrix operation.
 *
 */
public class SumMatrixOperation extends AbstractMatrixOperation {

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
     * Constructor for sum matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     */
    public SumMatrixOperation(int rows, int columns) {
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
        this.value += value;
        count++;
    }

    /**
     * Applies sum operation.
     *
     * @param input input matrix.
     * @return sum of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double applySum(Matrix input) throws MatrixException {
        this.input = input;
        applyMatrixOperation();
        return value;
    }

    /**
     * Applies mean operation.
     *
     * @param input input matrix.
     * @return mean of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double applyMean(Matrix input) throws MatrixException {
        this.input = input;
        applyMatrixOperation();
        return value / (double)count;
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
