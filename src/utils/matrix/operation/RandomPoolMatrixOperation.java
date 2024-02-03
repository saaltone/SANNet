/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;

import java.util.Random;

/**
 * Implements random pooling matrix operation.<br>
 * Selects each input of pool for propagation randomly with uniform probability.<br>
 *
 */
public class RandomPoolMatrixOperation extends AbstractPositionalPoolingMatrixOperation {

    /**
     * Current input row.
     *
     */
    private int randomRow;

    /**
     * Current input column.
     *
     */
    private int randomColumn;

    /**
     * Random number generator.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for random pooling matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param filterRowSize filter size in rows.
     * @param filterColumnSize filter size in columns.
     * @param dilation dilation step
     * @param stride stride step
     */
    public RandomPoolMatrixOperation(int rows, int columns, int depth, int filterRowSize, int filterColumnSize, int dilation, int stride) {
        super(rows, columns, depth, filterRowSize, filterColumnSize, dilation, stride);
    }

    /**
     * Applies convolution operation.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     * @param inputRow current input row.
     * @param inputColumn current input column.
     * @param filterRow current filter row.
     * @param filterColumn current filter column.
     * @param value current value.
     * @param result result matrix.
     */
    protected void applyOperation(int row, int column, int depth, int inputRow, int inputColumn, int filterRow, int filterColumn, double value, Matrix result) {
    }

    /**
     * Applies masked convolution operation.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     * @param inputRow current input row.
     * @param inputColumn current input column.
     * @param filterRow current filter row.
     * @param filterColumn current filter column.
     * @param value current value.
     * @param result result matrix.
     */
    protected void applyMaskOperation(int row, int column, int depth, int inputRow, int inputColumn, int filterRow, int filterColumn, double value, Matrix result) {
        while (hasMaskAt(randomRow, randomColumn, depth, getFirst())) {
            startOperation(row, column, depth);
        }
    }

    /**
     * Starts convolutional operation
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    protected void startOperation(int row, int column, int depth) {
        randomRow = row + random.nextInt(getFilterRows());
        randomColumn = column + random.nextInt(getFilterColumns());
    }

    /**
     * Finishes convolutional operation
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     * @param result result matrix.
     */
    protected void finishOperation(int row, int column, int depth, Matrix result) {
        result.setValue(row, column, depth, getFirst().getValue(randomRow, randomColumn, depth));
        super.finishOperation(row, column, depth);
    }

    /**
     * Returns input row.
     *
     * @return input row.
     */
    protected int getInputRow() {
        return randomRow;
    }

    /**
     * Returns input column.
     *
     * @return input column.
     */
    protected int getInputColumn() {
        return randomColumn;
    }

}
