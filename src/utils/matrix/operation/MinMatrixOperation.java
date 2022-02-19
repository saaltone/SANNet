/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements minimum matrix operation.
 *
 */
public class MinMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input matrix.
     *
     */
    private Matrix input;

    /**
     * Minimum value.
     *
     */
    private double minValue = Double.POSITIVE_INFINITY;

    /**
     * Minimum row.
     *
     */
    private int minRow = -1;

    /**
     * Minimum column.
     *
     */
    private int minColumn = -1;

    /**
     * Constructor for min matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     */
    public MinMatrixOperation(int rows, int columns) {
        super(rows, columns, true);
    }

    /**
     * Applies argmin operation.
     *
     * @param input input matrix.
     * @return minimum arguments (row and column)
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public int[] applyArgMin(Matrix input) throws MatrixException {
        this.input = input;
        applyMatrixOperation();
        int[] result = new int[2];
        result[0] = getMinRow();
        result[1] = getMinColumn();
        return result;
    }

    /**
     * Applies minimum operation.
     *
     * @param input input matrix.
     * @return minimum value
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double applyMin(Matrix input) throws MatrixException {
        this.input = input;
        applyMatrixOperation();
        return getMinValue();
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
        if (value < this.minValue) {
            this.minValue = value;
            this.minRow = row;
            this.minColumn = column;
        }
    }

    /**
     * Returns min value;
     *
     * @return min value;
     */
    public double getMinValue() {
        return minValue;
    }

    /**
     * Returns min row.
     *
     * @return min row.
     */
    public int getMinRow() {
        return minRow;
    }

    /**
     * Returns min column.
     *
     * @return min column.
     */
    public int getMinColumn() {
        return minColumn;
    }

}
