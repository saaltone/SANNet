/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
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
     * Minimum value.
     *
     */
    private transient double minValue = Double.MAX_VALUE;

    /**
     * Minimum row.
     *
     */
    private transient int minRow = -1;

    /**
     * Minimum column.
     *
     */
    private transient int minColumn = -1;

    /**
     * Minimum depth.
     *
     */
    private transient int minDepth = -1;

    /**
     * Constructor for min matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     */
    public MinMatrixOperation(int rows, int columns, int depth) {
        super(rows, columns, depth, true);
    }

    /**
     * Applies argmin operation.
     *
     * @param first first matrix.
     * @return minimum arguments (row and column)
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public int[] applyArgMin(Matrix first) throws MatrixException {
        minValue = Double.MAX_VALUE;
        minRow = -1;
        minColumn = -1;
        minDepth = -1;
        applyMatrixOperation(first, null, null);
        int[] result = new int[3];
        result[0] = getMinRow();
        result[1] = getMinColumn();
        result[2] = getMinDepth();
        return result;
    }

    /**
     * Applies minimum operation.
     *
     * @param first first matrix.
     * @return minimum value
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double applyMin(Matrix first) throws MatrixException {
        applyMatrixOperation(first, null, null);
        return getMinValue();
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
        if (minValue > value || minValue == Double.MAX_VALUE) {
            minValue = value;
            minRow = row;
            minColumn = column;
            minDepth = depth;
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

    /**
     * Returns min depth.
     *
     * @return min depth.
     */
    public int getMinDepth() {
        return minDepth;
    }

}
