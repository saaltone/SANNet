/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;

/**
 * Implements abstract convolutional operation.
 *
 */
public abstract class AbstractConvolutionalOperation extends AbstractMatrixOperation {

    /**
     * Target matrix.
     *
     */
    private transient Matrix targetMatrix;

    /**
     * Result matrix.
     *
     */
    private transient Matrix result;

    /**
     * Input gradient row size.
     *
     */
    private final int inputRows;

    /**
     * Input gradient column size.
     *
     */
    private final int inputColumns;

    /**
     * Input depth.
     *
     */
    protected final int inputDepth;

    /**
     * Filter row size.
     *
     */
    private final int filterRowSize;

    /**
     * Filter column size.
     *
     */
    private final int filterColumnSize;

    /**
     * Matrix dilation value.
     *
     */
    private final int dilation;

    /**
     * Constructor for abstract convolution operation.
     *
     * @param rows             number of rows for operation.
     * @param columns          number of columns for operation.
     * @param depth            depth for operation.
     * @param inputDepth       input depth.
     * @param filterRowSize    filter row size
     * @param filterColumnSize filter column size.
     * @param dilation         dilation step
     * @param stride           stride step
     * @param provideValue if true operation provides value when applying operation otherwise false.
     */
    public AbstractConvolutionalOperation(int rows, int columns, int depth, int inputDepth, int filterRowSize, int filterColumnSize, int dilation, int stride, boolean provideValue) {
        super(rows, columns, depth, provideValue, stride);
        this.inputRows = rows + filterRowSize - 1;
        this.inputColumns = columns + filterColumnSize - 1;
        this.inputDepth = inputDepth;
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
        this.dilation = dilation;
    }

    /**
     * Sets target matrix.
     *
     * @param targetMatrix target matrix.
     */
    protected void setTargetMatrix(Matrix targetMatrix) {
        this.targetMatrix = targetMatrix;
    }

    /**
     * Returns target matrix.
     *
     * @return target matrix.
     */
    protected Matrix getTargetMatrix() {
        return targetMatrix;
    }

    /**
     * Sets result matrix.
     *
     * @param result result matrix.
     */
    protected void setResult(Matrix result) {
        this.result = result;
    }

    /**
     * Returns result matrix.
     *
     * @return result matrix.
     */
    protected Matrix getResult() {
        return result;
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
     * Returns input rows.
     *
     * @return input rows.
     */
    protected int getInputRows() {
        return inputRows;
    }

    /**
     * Returns input columns.
     *
     * @return input columns.
     */
    protected int getInputColumns() {
        return inputColumns;
    }

    /**
     * Returns input depth.
     *
     * @return input depth.
     */
    protected int getInputDepth() {
        return inputDepth;
    }

    /**
     * Returns number of filter rows.
     *
     * @return number of filter rows.
     */
    protected int getFilterRows() {
        return filterRowSize;
    }

    /**
     * Returns number of filter columns.
     *
     * @return number of filter columns.
     */
    protected int getFilterColumns() {
        return filterColumnSize;
    }

    /**
     * Returns dilation.
     *
     * @return dilation.
     */
    private int getDilation() {
        return dilation;
    }

    /**
     * Returns filter row.
     *
     * @param filterRow filter row.
     * @return filter row.
     */
    protected int getFilterRow(int filterRow) {
        return filterRow;
    }

    /**
     * Returns filter column.
     *
     * @param filterColumn filter column.
     * @return filter column.
     */
    protected int getFilterColumn(int filterColumn) {
        return filterColumn;
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
        apply(row, column, depth, value, false);
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
    protected abstract void applyOperation(int row, int column, int depth, int inputRow, int inputColumn, int filterRow, int filterColumn, double value);

    /**
     * Applies operation assuming masked matrices.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     * @param value current value.
     */
    public void applyMask(int row, int column, int depth, double value) {
        apply(row, column, depth, value, true);
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
    protected abstract void applyMaskOperation(int row, int column, int depth, int inputRow, int inputColumn, int filterRow, int filterColumn, double value);

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     * @param value current value.
     * @param asMasked if true applied operation as masked otherwise as non-masked.
     */
    private void apply(int row, int column, int depth, double value, boolean asMasked) {
        startOperation(row, column, depth);
        for (int filterRow = 0; filterRow < getFilterRows(); filterRow += getDilation()) {
            for (int filterColumn = 0; filterColumn < getFilterColumns(); filterColumn += getDilation()) {
                int currentFilterRow = getFilterRow(filterRow);
                int currentFilterColumn = getFilterColumn(filterColumn);
                int inputRow = getCurrentInputRow(row, currentFilterRow);
                int inputColumn = getCurrentInputColumn(column, currentFilterColumn);
                if (isValidInputPosition(inputRow, inputColumn)) {
                    if (asMasked) applyMaskOperation(row, column, depth, inputRow, inputColumn, currentFilterRow, currentFilterColumn, value);
                    else applyOperation(row, column, depth, inputRow, inputColumn, currentFilterRow, currentFilterColumn, value);
                }
            }
        }
        finishOperation(row, column, depth);
    }

    /**
     * Starts convolutional operation
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    protected abstract void startOperation(int row, int column, int depth);

    /**
     * Finishes convolutional operation
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    protected abstract void finishOperation(int row, int column, int depth);

    /**
     * Returns current input row.
     *
     * @param row row
     * @param filterRow filter row
     * @return current input row.
     */
    protected int getCurrentInputRow(int row, int filterRow) {
        return row + filterRow;
    }

    /**
     * Returns current input column.
     *
     * @param column column
     * @param filterColumn filter column
     * @return current input column.
     */
    protected int getCurrentInputColumn(int column, int filterColumn) {
        return column + filterColumn;
    }

    /**
     * Checks if input row and columns are valid.
     *
     * @param inputRow input row
     * @param inputColumn input column
     * @return true if input row and column are valid otherwise returns false.
     */
    protected boolean isValidInputPosition(int inputRow, int inputColumn) {
        return (inputRow >= 0 && inputColumn >= 0 && inputRow < getInputRows() && inputColumn < getInputColumns());
    }

}
