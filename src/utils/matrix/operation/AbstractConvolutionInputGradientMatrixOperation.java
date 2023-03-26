/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements abstract convolution input gradient matrix operation.
 *
 */
public abstract class AbstractConvolutionInputGradientMatrixOperation extends AbstractMatrixOperation {

    /**
     * Output gradient.
     *
     */
    protected Matrix outputGradient;

    /**
     * Filter matrix.
     *
     */
    protected Matrix filter;

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
     * Resulting input gradient.
     *
     */
    protected Matrix inputGradient;

    /**
     * Total input depth.
     *
     */
    protected int totalInputDepth;

    /**
     * If true convolution is depth separable
     *
     */
    protected final boolean isDepthSeparable;

    /**
     * Constructor for abstract convolution input gradient matrix operation.
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
    public AbstractConvolutionInputGradientMatrixOperation(int rows, int columns, int depth, int filterRowSize, int filterColumnSize, int dilation, int stride, boolean isDepthSeparable) {
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
     * @param filter filter matrix.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix outputGradient, Matrix filter) throws MatrixException {
        this.outputGradient = outputGradient;
        this.filter = filter;
        this.inputGradient = outputGradient.getNewMatrix(getRows() - filterRowSize + 1, getColumns() - filterColumnSize + 1, getDepth());
        this.totalInputDepth = inputGradient.getDepth();
        applyMatrixOperation();
        return inputGradient;
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
                double filterValue = filter.getValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth) * value;
                if (isDepthSeparable) {
                    inputGradient.addByValue(row + filterRow, column + filterColumn, depth, filterValue * value);
                }
                else {
                    for (int inputDepth = 0; inputDepth < totalInputDepth; inputDepth++) {
                        inputGradient.addByValue(row + filterRow, column + filterColumn, inputDepth, filterValue * value);
                    }
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
                double gradientValue = filter.getValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth) * value;
                if (isDepthSeparable) {
                    inputGradient.addByValue(row + filterRow, column + filterColumn, depth, gradientValue);
                }
                else {
                    if (!hasMaskAt(filterRow, filterColumn, depth, filter)) {
                        for (int inputDepth = 0; inputDepth < totalInputDepth; inputDepth++) {
                            inputGradient.addByValue(row + filterRow, column + filterColumn, inputDepth, gradientValue);
                        }
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
