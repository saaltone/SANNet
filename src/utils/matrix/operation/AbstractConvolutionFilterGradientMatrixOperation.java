/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements abstract convolution filter gradient matrix operation.
 *
 */
public abstract class AbstractConvolutionFilterGradientMatrixOperation extends AbstractMatrixOperation {

    /**
     * Output gradient.
     *
     */
    protected Matrix outputGradient;

    /**
     * Input matrix.
     *
     */
    protected Matrix input;

    /**
     * Total input depth.
     *
     */
    protected int totalInputDepth;

    /**
     * Number of rows in filter.
     *
     */
    protected final int filterRowSize;

    /**
     * Number of columns in filter.
     *
     */
    protected final int filterColumnSize;

    /**
     * Dilation.
     *
     */
    protected final int dilation;

    /**
     * If true convolution is depth separable
     *
     */
    protected final boolean isDepthSeparable;

    /**
     * Resulting filter gradient.
     *
     */
    protected Matrix filterGradient;

    /**
     * Constructor for abstract convolution filter gradient matrix operation.
     *
     * @param rows             number of rows for operation.
     * @param columns          number of columns for operation.
     * @param depth            depth for operation.
     * @param filterRowSize    filter row size
     * @param filterColumnSize filter column size.
     * @param dilation         dilation step
     * @param stride           stride step
     * @param isDepthSeparable if true convolution is depth separable
     */
    public AbstractConvolutionFilterGradientMatrixOperation(int rows, int columns, int depth, int filterRowSize, int filterColumnSize, int dilation, int stride, boolean isDepthSeparable) {
        super(rows, columns, depth, true, stride);
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
        this.dilation = dilation;
        this.isDepthSeparable = isDepthSeparable;
    }

    /**
     * Applies matrix operation.
     *
     * @param outputGradient output gradient.
     * @param input input matrix.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix outputGradient, Matrix input) throws MatrixException {
        this.outputGradient = outputGradient;
        this.input = input;
        this.totalInputDepth = input.getDepth();
        this.filterGradient = outputGradient.getNewMatrix(filterRowSize, filterColumnSize, getDepth());
        applyMatrixOperation();
        return filterGradient;
    }

    /**
     * Returns target matrix.
     *
     * @return target matrix.
     */
    protected Matrix getTargetMatrix() {
        return outputGradient;
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
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     * @param value current value.
     */
    public void apply(int row, int column, int depth, double value) {
        for (int filterRow = 0; filterRow < filterRowSize; filterRow += dilation) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn += dilation) {
                if (isDepthSeparable) {
                    double inputValue = input.getValue(row + filterRow, column + filterColumn, depth);
                    filterGradient.addByValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth, inputValue * value);
                }
                else {
                    double sumValue = 0;
                    for (int inputDepth = 0; inputDepth < totalInputDepth; inputDepth++) {
                        double inputValue = input.getValue(row + filterRow, column + filterColumn, inputDepth);
                        sumValue += inputValue * value;
                    }
                    filterGradient.addByValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth, sumValue);
                }
            }
        }
    }

    /**
     * Applies operation assuming masked matrices.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     * @param value current value.
     */
    public void applyMask(int row, int column, int depth, double value) {
        for (int filterRow = 0; filterRow < filterRowSize; filterRow += dilation) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn += dilation) {
                if (isDepthSeparable) {
                    if (!hasMaskAt(row + filterRow, column + filterColumn, depth, input)) {
                        double inputValue = input.getValue(row + filterRow, column + filterColumn, depth);
                        filterGradient.addByValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth, inputValue * value);
                    }
                }
                else {
                    double sumValue = 0;
                    for (int inputDepth = 0; inputDepth < totalInputDepth; inputDepth++) {
                        if (!hasMaskAt(row + filterRow, column + filterColumn, inputDepth, input)) {
                            double inputValue = input.getValue(row + filterRow, column + filterColumn, depth);
                            sumValue += inputValue * value;
                        }
                    }
                    filterGradient.addByValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth, sumValue);
                }
            }
        }
    }

    /**
     * Returns filter row.
     *
     * @param filterRow filter row.
     * @return filter row.
     */
    protected abstract int getFilterRow(int filterRow);

    /**
     * Returns filter column.
     *
     * @param filterColumn filter column.
     * @return filter column.
     */
    protected abstract int getFilterColumn(int filterColumn);

}
