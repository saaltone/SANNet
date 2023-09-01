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
public abstract class AbstractConvolutionOperation extends AbstractMatrixOperation {

    /**
     * Input matrix.
     *
     */
    private Matrix inputMatrix;

    /**
     * Target matrix.
     *
     */
    private Matrix targetMatrix;

    /**
     * Result matrix.
     *
     */
    private Matrix result;

    /**
     * Filter row size.
     *
     */
    protected final int filterRowSize;

    /**
     * Filter column size.
     *
     */
    protected final int filterColumnSize;

    /**
     * Matrix dilation value.
     *
     */
    protected final int dilation;

    /**
     * If true convolution is depth separable
     *
     */
    protected final boolean isDepthSeparable;

    /**
     * If true operation is executed as convolution otherwise as crosscorrelation
     *
     */
    protected final boolean asConvolution;

    /**
     * Constructor for abstract convolution operation.
     *
     * @param rows             number of rows for operation.
     * @param columns          number of columns for operation.
     * @param depth            depth for operation.
     * @param filterRowSize    filter row size
     * @param filterColumnSize filter column size.
     * @param dilation         dilation step
     * @param stride           stride step
     * @param isDepthSeparable if true convolution is depth separable
     * @param asConvolution    if true operation is executed as convolution otherwise as crosscorrelation
     */
    public AbstractConvolutionOperation(int rows, int columns, int depth, int filterRowSize, int filterColumnSize, int dilation, int stride, boolean isDepthSeparable, boolean asConvolution) {
        super(rows, columns, depth, false, stride);
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
        this.dilation = dilation;
        this.isDepthSeparable = isDepthSeparable;
        this.asConvolution = asConvolution && (filterRowSize > 1 || filterColumnSize > 1);
    }

    /**
     * Sets input matrix.
     *
     * @param inputMatrix input matrix.
     */
    protected void setInputMatrix(Matrix inputMatrix) {
        this.inputMatrix = inputMatrix;
    }

    /**
     * Returns input matrix.
     *
     * @return input matrix.
     */
    protected Matrix getInputMatrix() {
        return inputMatrix;
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
     * Returns filter row.
     *
     * @param filterRow filter row.
     * @return filter row.
     */
    protected int getFilterRow(int filterRow) {
        return asConvolution ? filterRowSize - 1 - filterRow * dilation : filterRow * dilation;
    }

    /**
     * Returns filter column.
     *
     * @param filterColumn filter column.
     * @return filter column.
     */
    protected int getFilterColumn(int filterColumn) {
        return asConvolution ? filterColumnSize - 1 - filterColumn * dilation : filterColumn * dilation;
    }

}
