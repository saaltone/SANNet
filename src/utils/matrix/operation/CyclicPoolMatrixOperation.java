/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

/**
 * Implements cyclic pooling matrix operation.<br>
 * Traverses cyclically each filter row and column through step by step and propagates selected row and column.<br>
 *
 */
public class CyclicPoolMatrixOperation extends AbstractPositionalPoolingMatrixOperation {

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
     * Current row of filter.
     *
     */
    private transient int currentRow = 0;

    /**
     * Current column of filter.
     *
     */
    private transient int currentColumn = 0;

    /**
     * Constructor for cyclic pooling matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param filterRowSize filter size in rows.
     * @param filterColumnSize filter size in columns.
     * @param dilation dilation step
     * @param stride stride step
     */
    public CyclicPoolMatrixOperation(int rows, int columns, int depth, int filterRowSize, int filterColumnSize, int dilation, int stride) {
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
     */
    protected void applyOperation(int row, int column, int depth, int inputRow, int inputColumn, int filterRow, int filterColumn, double value) {
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
     */
    protected void applyMaskOperation(int row, int column, int depth, int inputRow, int inputColumn, int filterRow, int filterColumn, double value) {
        while (hasMaskAt(currentRow, currentColumn, depth, getTargetMatrix())) {
            if(++currentRow >= getFilterRows()) {
                currentRow = 0;
                if(++currentColumn >= getFilterColumns()) currentColumn = 0;
            }
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
        inputRow = row + currentRow;
        inputColumn = column + currentColumn;
    }

    /**
     * Finishes convolutional operation
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    protected void finishOperation(int row, int column, int depth) {
        getResult().setValue(row, column, depth, getTargetMatrix().getValue(inputRow, inputColumn, depth));

        super.finishOperation(row, column, depth);

        if(++currentRow >= getFilterRows()) {
            currentRow = 0;
            if(++currentColumn >= getFilterColumns()) currentColumn = 0;
        }
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
