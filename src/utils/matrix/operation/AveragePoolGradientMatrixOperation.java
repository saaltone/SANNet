/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements average pooling gradient matrix operation.
 *
 */
public class AveragePoolGradientMatrixOperation extends AbstractConvolutionalOperation {

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
     * Inverted size of filter 1 / (filterRowSize * filterColumnSize)
     *
     */
    private final double invertedFilterSize;

    /**
     * Constructor for average pooling gradient matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param filterRowSize filter size in rows.
     * @param filterColumnSize filter size in columns.
     * @param dilation dilation step
     * @param stride stride step
     */
    public AveragePoolGradientMatrixOperation(int rows, int columns, int depth, int filterRowSize, int filterColumnSize, int dilation, int stride) {
        super(rows, columns, depth, filterRowSize, filterColumnSize, dilation, stride, true);
        this.inputRows = rows + filterRowSize - 1;
        this.inputColumns = columns + filterColumnSize - 1;
        this.invertedFilterSize = 1 / (double)(filterRowSize * filterColumnSize);
    }

    /**
     * Applies matrix operation.
     *
     * @param outputGradient output gradient.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix outputGradient) throws MatrixException {
        setTargetMatrix(outputGradient);
        setResult(outputGradient.getNewMatrix(inputRows, inputColumns, getDepth()));
        applyMatrixOperation();
        return getResult();
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
        double gradientValue = value * invertedFilterSize;
        getResult().addByValue(inputRow, inputColumn, depth, gradientValue);
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
        applyOperation(row, column, depth, inputRow, inputColumn, filterRow, filterColumn, value);
    }

    /**
     * Starts convolutional operation
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    protected void startOperation(int row, int column, int depth) {
    }

    /**
     * Finishes convolutional operation
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    protected void finishOperation(int row, int column, int depth) {
    }

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
        return (inputRow >= 0 && inputColumn >= 0 && inputRow < inputRows && inputColumn < inputColumns);
    }

}
