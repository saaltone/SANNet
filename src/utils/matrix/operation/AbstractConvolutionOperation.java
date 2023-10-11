/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;

/**
 * Implements abstract convolution operation.
 *
 */
public abstract class AbstractConvolutionOperation extends AbstractConvolutionalOperation {

    /**
     * If true convolution is depth separable
     *
     */
    private final boolean isDepthSeparable;

    /**
     * If true operation is executed as convolution otherwise as crosscorrelation
     *
     */
    private final boolean asConvolution;

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
     * @param isDepthSeparable if true convolution is depth separable
     * @param asConvolution    if true operation is executed as convolution otherwise as crosscorrelation
     * @param provideValue if true operation provides value when applying operation otherwise false.
     */
    public AbstractConvolutionOperation(int rows, int columns, int depth, int inputDepth, int filterRowSize, int filterColumnSize, int dilation, int stride, boolean isDepthSeparable, boolean asConvolution, boolean provideValue) {
        super(rows, columns, depth, inputDepth, filterRowSize, filterColumnSize, dilation, stride, provideValue);
        this.isDepthSeparable = isDepthSeparable;
        this.asConvolution = asConvolution && (filterRowSize > 1 || filterColumnSize > 1);
    }

    /**
     * Returns if convolution is depth separable.
     *
     * @return true if convolution is depth separable otherwise false.
     */
    protected boolean getIsDepthSeparable() {
        return isDepthSeparable;
    }

    /**
     * Returns filter row.
     *
     * @param filterRow filter row.
     * @return filter row.
     */
    protected int getFilterRow(int filterRow) {
        return asConvolution ? getFilterRows() - 1 - filterRow : filterRow;
    }

    /**
     * Returns filter column.
     *
     * @param filterColumn filter column.
     * @return filter column.
     */
    protected int getFilterColumn(int filterColumn) {
        return asConvolution ? getFilterColumns() - 1 - filterColumn : filterColumn;
    }

    /**
     * Returns filter position based on combined position of input depth and filter
     *
     * @param inputDepth input depth
     * @param filter filter
     * @return filter position
     */
    protected int getFilterPosition(int inputDepth, int filter) {
        return getDepth() * inputDepth + filter;
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
     * @param result result matrix.
     */
    protected void finishOperation(int row, int column, int depth, Matrix result) {
    }

}
