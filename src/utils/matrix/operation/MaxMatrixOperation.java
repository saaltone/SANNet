/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements maximum matrix operation.
 *
 */
public class MaxMatrixOperation extends AbstractMatrixOperation {

    /**
     * Maximum value.
     *
     */
    private double maxValue = Double.MIN_VALUE;

    /**
     * Maximum row.
     *
     */
    private transient int maxRow = -1;

    /**
     * Maximum column.
     *
     */
    private transient int maxColumn = -1;

    /**
     * Maximum depth.
     *
     */
    private transient int maxDepth = -1;

    /**
     * Constructor for max matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     */
    public MaxMatrixOperation(int rows, int columns, int depth) {
        super(rows, columns, depth, true);
    }

    /**
     * Applies argmax operation.
     *
     * @param first first matrix.
     * @return maximum arguments (row and column)
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public int[] applyArgMax(Matrix first) throws MatrixException {
        maxValue = Double.MIN_VALUE;
        maxRow = -1;
        maxColumn = -1;
        maxDepth = -1;
        applyMatrixOperation(first, null, null);
        int[] result = new int[3];
        result[0] = getMaxRow();
        result[1] = getMaxColumn();
        result[2] = getMaxDepth();
        return result;
    }

    /**
     * Applies maximum operation.
     *
     * @param first first matrix.
     * @return maximum value
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double applyMax(Matrix first) throws MatrixException {
        applyMatrixOperation(first, null, null);
        return getMaxValue();
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
        if (maxValue < value || maxValue == Double.MIN_VALUE) {
            maxValue = value;
            maxRow = row;
            maxColumn = column;
            maxDepth = depth;
        }
    }

    /**
     * Returns max value;
     *
     * @return max value;
     */
    public double getMaxValue() {
        return maxValue;
    }

    /**
     * Returns max row.
     *
     * @return max row.
     */
    public int getMaxRow() {
        return maxRow;
    }

    /**
     * Returns max column.
     *
     * @return max column.
     */
    public int getMaxColumn() {
        return maxColumn;
    }

    /**
     * Returns max depth.
     *
     * @return max depth.
     */
    public int getMaxDepth() {
        return maxDepth;
    }

}
