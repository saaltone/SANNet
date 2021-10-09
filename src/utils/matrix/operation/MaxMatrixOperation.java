/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Defines maximum matrix operation.
 *
 */
public class MaxMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input matrix.
     *
     */
    private Matrix input;

    /**
     * Maximum value.
     *
     */
    private double value = Double.NEGATIVE_INFINITY;

    /**
     * Maximum row.
     *
     */
    private int row = -1;

    /**
     * Maximum column.
     *
     */
    private int column = -1;

    /**
     * Constructor for max matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     */
    public MaxMatrixOperation(int rows, int columns) {
        super(rows, columns, true);
    }

    /**
     * Applies argmax operation.
     *
     * @param input input matrix.
     * @return maximum arguments (row and column)
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public int[] applyArgMax(Matrix input) throws MatrixException {
        this.input = input;
        applyMatrixOperation();
        int[] result = new int[2];
        result[0] = getRow();
        result[1] = getColumn();
        return result;
    }

    /**
     * Applies maximum operation.
     *
     * @param input input matrix.
     * @return maximum value
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double applyMax(Matrix input) throws MatrixException {
        this.input = input;
        applyMatrixOperation();
        return getValue();
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
        if (value > this.value) {
            this.value = value;
            this.row = row;
            this.column = column;
        }
    }

    /**
     * Returns min value;
     *
     * @return min value;
     */
    public double getValue() {
        return value;
    }

    /**
     * Returns min row.
     *
     * @return min row.
     */
    public int getRow() {
        return row;
    }

    /**
     * Returns min column.
     *
     * @return min column.
     */
    public int getColumn() {
        return column;
    }

}
