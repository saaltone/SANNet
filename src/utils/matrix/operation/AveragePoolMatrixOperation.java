/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
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
     * @param input input matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix input) throws MatrixException {
        setTargetMatrix(input);
        setResult(input.getNewMatrix(getRows(), getColumns(), getDepth()));
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
        double filterValue = getTargetMatrix().getValue(inputRow, inputColumn, depth);
        sumValue += filterValue;
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
        if (!hasMaskAt(inputRow, inputColumn, depth, getTargetMatrix())) {
            applyOperation(row, column, depth, inputRow, inputColumn, filterRow, filterColumn, value);
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
        sumValue = 0;
    }

    /**
     * Finishes convolutional operation
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    protected void finishOperation(int row, int column, int depth) {
        getResult().setValue(row, column, depth, sumValue * invertedFilterSize);
    }

}
