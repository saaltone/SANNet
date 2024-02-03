/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements average pooling matrix operation.
 *
 */
public class AveragePoolMatrixOperation extends AbstractConvolutionalOperation {

    /**
     * First matrix.
     *
     */
    private transient Matrix first;

    /**
     * Inverted size of filter = 1 / (rows * columns)
     *
     */
    private final double invertedFilterSize;

    /**
     * Sum value
     *
     */
    private transient double sumValue = 0;

    /**
     * Constructor for average pooling matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param filterRowSize filter size in rows.
     * @param filterColumnSize filter size in columns.
     * @param dilation dilation step
     * @param stride stride step
     */
    public AveragePoolMatrixOperation(int rows, int columns, int depth, int filterRowSize, int filterColumnSize, int dilation, int stride) {
        super(rows, columns, depth, depth, filterRowSize, filterColumnSize, dilation, stride, false);
        this.invertedFilterSize = 1 / (double)(filterRowSize * filterColumnSize);
    }

    /**
     * Applies matrix operation.
     *
     * @param first first matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix first) throws MatrixException {
        this.first = first;
        return applyMatrixOperation(first, null, first.getNewMatrix(getRows(), getColumns(), getDepth()));
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
        sumValue += first.getValue(inputRow, inputColumn, depth);
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
        applyOperation(row, column, depth, inputRow, inputColumn, filterRow, filterColumn, value, result);
    }

    /**
     * Starts convolutional operation
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    protected void startOperation(int row, int column, int depth) {
        sumValue = 0;
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
        result.setValue(row, column, depth, sumValue * invertedFilterSize);
    }

}
