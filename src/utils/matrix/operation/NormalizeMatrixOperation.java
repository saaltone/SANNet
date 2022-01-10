/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Defines normalize matrix operation.
 *
 */
public class NormalizeMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input matrix.
     *
     */
    private Matrix input;

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
     * Normalized matrix.
     *
     */
    private Matrix result;

    /**
     * Constructor for normalize matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param mean mean for normalize operation.
     * @param variance variance for normalize operation.
     */
    public NormalizeMatrixOperation(int rows, int columns, double mean, double variance) {
        super(rows, columns, true);
        this.mean = mean;
        this.variance = variance;
    }

    /**
     * Applies operation.
     *
     * @param input input matrix.
     * @param result result matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix input, Matrix result) throws MatrixException {
        this.input = input;
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
        return input;
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
     * Sets result matrix.
     *
     * @param result result matrix.
     */
    public void setResult(Matrix result) {
        this.result = result;
    }

    /**
     * Returns result matrix.
     *
     * @return result matrix.
     */
    public Matrix getResult() {
        return result;
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     */
    public void apply(int row, int column, double value) {
        result.setValue(row, column, (value - mean) / variance);
    }

}
