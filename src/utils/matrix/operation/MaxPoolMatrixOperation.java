/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

/**
 * Implements max pooling matrix operation.
 *
 */
public class MaxPoolMatrixOperation extends AbstractPositionalPoolingMatrixOperation {

    /**
     * Current input row.
     *
     */
    private int inputRow;

    /**
     * Current input column.
     *
     */
    private int inputColumn;

    /**
     * Constructor for max pooling matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param inputRowSize number of input rows.
     * @param inputColumnSize number of input columns.
     * @param filterRowSize filter size in rows.
     * @param filterColumnSize filter size in columns.
     * @param stride stride step
     */
    public MaxPoolMatrixOperation(int rows, int columns, int depth, int inputRowSize, int inputColumnSize, int filterRowSize, int filterColumnSize, int stride) {
        super(rows, columns, depth, inputRowSize, inputColumnSize, filterRowSize, filterColumnSize, stride);
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    protected void executeApply(int row, int column, int depth) {
        inputRow = -1;
        inputColumn = -1;
        double maxValue = Double.MIN_VALUE;
        for (int filterRow = 0; filterRow < filterRowSize; filterRow++) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn++) {
                int currentRow = row + filterRow;
                int currentColumn = column + filterColumn;
                double filterValue = input.getValue(currentRow, currentColumn, depth);
                if (maxValue < filterValue) {
                    maxValue = filterValue;
                    inputRow = currentRow;
                    inputColumn = currentColumn;
                }
            }
        }
        result.setValue(row, column, depth, maxValue);
    }

    /**
     * Applies operation assuming masked matrices.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    public void executeApplyMask(int row, int column, int depth) {
        inputRow = -1;
        inputColumn = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int filterRow = 0; filterRow < filterRowSize; filterRow++) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn++) {
                int currentRow = row + filterRow;
                int currentColumn = column + filterColumn;
                if (!hasMaskAt(currentRow, currentColumn, depth, input)) {
                    double inputValue = input.getValue(currentRow, currentColumn, depth);
                    if (maxValue < inputValue) {
                        maxValue = inputValue;
                        inputRow = currentRow;
                        inputColumn = currentColumn;
                    }
                }
            }
        }
        result.setValue(row, column, depth, maxValue);
    }

    /**
     * Returns input row.
     *
     * @return input row.
     */
    protected int getInputRow() {
        return inputRow;
    }

    /**
     * Returns input column.
     *
     * @return input column.
     */
    protected int getInputColumn() {
        return inputColumn;
    }

}
