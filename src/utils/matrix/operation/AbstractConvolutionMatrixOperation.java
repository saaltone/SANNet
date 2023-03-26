/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements abstract convolution matrix operation.
 *
 */
public abstract class AbstractConvolutionMatrixOperation extends AbstractMatrixOperation {

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
     * Filter matrix.
     *
     */
    protected Matrix filter;

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
     * Result matrix.
     *
     */
    protected Matrix result;

    /**
     * Constructor for abstract convolution matrix operation.
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
    public AbstractConvolutionMatrixOperation(int rows, int columns, int depth, int filterRowSize, int filterColumnSize, int dilation, int stride, boolean isDepthSeparable) {
        super(rows, columns, depth, false, stride);
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
        this.dilation = dilation;
        this.isDepthSeparable = isDepthSeparable;
    }

    /**
     * Applies matrix operation.
     *
     * @param input input matrix.
     * @param filter filter matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix input, Matrix filter) throws MatrixException {
        this.input = input;
        this.totalInputDepth = input.getDepth();
        this.filter = filter;
        this.result = input.getNewMatrix(getRows(), getColumns(), getDepth());
        applyMatrixOperation();
        return result;
    }

    /**
     * Returns target matrix.
     *
     * @return target matrix.
     */
    protected Matrix getTargetMatrix() {
        return input;
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
                double filterValue = filter.getValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth);
                if (isDepthSeparable) {
                    result.setValue(row, column, depth, input.getValue(row + filterRow, column + filterColumn, depth) * filterValue);
                }
                else {
                    double sumValue = 0;
                    for (int inputDepth = 0; inputDepth < totalInputDepth; inputDepth++) {
                        sumValue += input.getValue(row + filterRow, column + filterColumn, inputDepth);
                    }
                    result.setValue(row, column, depth, sumValue * filterValue);
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
                if (!hasMaskAt(filterRow, filterColumn, depth, filter)) {
                    double filterValue = filter.getValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth);
                    if (isDepthSeparable) {
                        if (!hasMaskAt(row + filterRow, column + filterColumn, depth, input)) {
                            result.setValue(row, column, depth, input.getValue(row + filterRow, column + filterColumn, depth) * filterValue);
                        }
                    }
                    else {
                        double sumValue = 0;
                        for (int inputDepth = 0; inputDepth < totalInputDepth; inputDepth++) {
                            if (!hasMaskAt(row + filterRow, column + filterColumn, inputDepth, input)) {
                                sumValue += input.getValue(row + filterRow, column + filterColumn, inputDepth);
                            }
                        }
                        result.setValue(row, column, depth, sumValue * filterValue);
                    }
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
