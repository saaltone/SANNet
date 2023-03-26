/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
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
     * Minimum depth.
     *
     */
    private int minDepth = -1;

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
     * @param input input matrix.
     * @return minimum arguments (row and column)
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public int[] applyArgMin(Matrix input) throws MatrixException {
        this.input = input;
        minValue = Double.POSITIVE_INFINITY;
        minRow = -1;
        minColumn = -1;
        minDepth = -1;
        applyMatrixOperation();
        int[] result = new int[3];
        result[0] = getMinRow();
        result[1] = getMinColumn();
        result[2] = getMinDepth();
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
        if (value < this.minValue) {
            this.minValue = value;
            this.minRow = row;
            this.minColumn = column;
            this.minDepth = depth;
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
