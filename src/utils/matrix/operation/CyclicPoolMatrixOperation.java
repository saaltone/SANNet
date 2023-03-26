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
     * @param inputRowSize number of input rows.
     * @param inputColumnSize number of input columns.
     * @param filterRowSize filter size in rows.
     * @param filterColumnSize filter size in columns.
     * @param stride stride step
     */
    public CyclicPoolMatrixOperation(int rows, int columns, int depth, int inputRowSize, int inputColumnSize, int filterRowSize, int filterColumnSize, int stride) {
        super(rows, columns, depth, inputRowSize, inputColumnSize, filterRowSize, filterColumnSize, stride);
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    public void executeApply(int row, int column, int depth) {
        inputRow = row + currentRow;
        inputColumn = column + currentColumn;
        result.setValue(row, column, depth, input.getValue(inputRow, inputColumn, depth));

        if(++currentRow >= filterRowSize) {
            currentRow = 0;
            if(++currentColumn >= filterColumnSize) currentColumn = 0;
        }
    }

    /**
     * Applies operation assuming masked matrices.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    public void executeApplyMask(int row, int column, int depth) {
        while (hasMaskAt(currentRow, currentColumn, 0, input)) {
            if(++currentRow >= filterRowSize) {
                currentRow = 0;
                if(++currentColumn >= filterColumnSize) currentColumn = 0;
            }
        }

        inputRow = row + currentRow;
        inputColumn = column + currentColumn;
        result.setValue(row, column, depth, input.getValue(inputRow, inputColumn, depth));

        if(++currentRow >= filterRowSize) {
            currentRow = 0;
            if(++currentColumn >= filterColumnSize) currentColumn = 0;
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
