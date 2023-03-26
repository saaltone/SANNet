/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements variance matrix operation.
 *
 */
public class VarianceMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input matrix.
     *
     */
    private Matrix input;

    /**
     * Mean value for variance operation.
     *
     */
    private final double mean;

    /**
     * Cumulated variance value.
     *
     */
    private double value;

    /**
     * Number of counted entries.
     *
     */
    private int count;

    /**
     * Constructor for variance operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param mean mean value for variance operation.
     */
    public VarianceMatrixOperation(int rows, int columns, int depth, double mean) {
        super(rows, columns, depth, true);
        this.mean = mean;
    }

    /**
     * Applies variance operation.
     *
     * @param input input matrix.
     * @return variance of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double applyVariance(Matrix input) throws MatrixException {
        this.input = input;
        value = 0;
        count = 0;
        applyMatrixOperation();
        return count > 0 ? value / (double)count : 0;
    }

    /**
     * Applies standard deviation operation.
     *
     * @param input input matrix.
     * @return standard deviation of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double applyStandardDeviation(Matrix input) throws MatrixException {
        this.input = input;
        applyMatrixOperation();
        return count > 1 ? Math.sqrt(value / (double)(count - 1)) : 0;
    }

    /**
     * Returns target matrix.
     *
     * @return target matrix.
     */
    protected Matrix getTargetMatrix() {
        return input;
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
        this.value += Math.pow(value - mean, 2);
        count++;
    }

}
