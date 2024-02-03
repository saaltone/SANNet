/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements abstract convolution matrix operation.
 *
 */
public abstract class AbstractConvolutionMatrixOperation extends AbstractConvolutionOperation {

    /**
     * First matrix.
     *
     */
    private transient Matrix first;

    /**
     * Filter matrix.
     *
     */
    private transient Matrix filter;

    /**
     * Value of convolution.
     *
     */
    private transient double convolutionValue = 0;

    /**
     * Constructor for abstract convolution matrix operation.
     *
     * @param rows             number of rows for operation.
     * @param columns          number of columns for operation.
     * @param depth            depth for operation.
     * @param inputDepth       input depth.
     * @param filterRowSize    filter row size
     * @param filterColumnSize filter column size.
     * @param dilation         dilation step
     * @param stride           stride step
     * @param isDepthSeparable if true convolution is depth separable
     * @param asConvolution    if true operation is executed as convolution otherwise as crosscorrelation
     */
    public AbstractConvolutionMatrixOperation(int rows, int columns, int depth, int inputDepth, int filterRowSize, int filterColumnSize, int dilation, int stride, boolean isDepthSeparable, boolean asConvolution) {
        super(rows, columns, depth, inputDepth, filterRowSize, filterColumnSize, dilation, stride, isDepthSeparable, asConvolution, false);
    }

    /**
     * Applies matrix operation.
     *
     * @param first first matrix.
     * @param filter filter matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix first, Matrix filter) throws MatrixException {
        this.first = first;
        this.filter = filter;
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
        if (getIsDepthSeparable()) {
            double inputValue = first.getValue(inputRow, inputColumn, depth);
            double filterValue = filter.getValue(filterRow, filterColumn, depth);
            convolutionValue += inputValue * filterValue;
        }
        else {
            for (int inputDepth = 0; inputDepth < getInputDepth(); inputDepth++) {
                double inputValue = first.getValue(inputRow, inputColumn, inputDepth);
                double filterValue = filter.getValue(filterRow, filterColumn, getFilterPosition(inputDepth, depth));
                convolutionValue += inputValue * filterValue;
            }
        }
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
        convolutionValue = 0;
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
        result.setValue(row, column, depth, convolutionValue);
    }

}
