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
     * Input matrix.
     *
     */
    private transient Matrix input;

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
     * Returns another matrix used in operation.
     *
     * @return another matrix used in operation.
     */
    public Matrix getOther() {
        return null;
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     * @param value current value.
     */
    public void apply(int row, int column, int depth, double value) {
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
        value = 0;
        count = 0;
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
