/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Defines norm matrix operation.
 *
 */
public class NormMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input matrix.
     *
     */
    private Matrix input;

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
     * @param p power for norm operation.
     */
    public NormMatrixOperation(int rows, int columns, int p) {
        super(rows, columns, true);
        this.p = p;
    }

    /**
     * Applies operation.
     *
     * @param input input matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double apply(Matrix input) throws MatrixException {
        this.input = input;
        applyMatrixOperation();
        return Math.pow(value, 1 / (double)p);
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
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     */
    public void apply(int row, int column, double value) {
        this.value += Math.pow(Math.abs(value), p);
    }

}
