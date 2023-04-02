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
     * Input gradient row size.
     *
     */
    private final int inputGradientRowSize;

    /**
     * Input gradient column size.
     *
     */
    private final int inputGradientColumnSize;

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
     * Input depth.
     *
     */
    protected final int inputDepth;

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
     * Constructor for abstract convolution input gradient matrix operation.
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
    public AbstractConvolutionInputGradientMatrixOperation(int rows, int columns, int depth, int inputDepth, int filterRowSize, int filterColumnSize, int dilation, int stride, boolean isDepthSeparable, boolean asConvolution) {
        super(rows, columns, depth, true, stride);
        this.inputDepth = inputDepth;
        this.inputGradientRowSize = rows + filterRowSize - 1;
        this.inputGradientColumnSize = columns + filterColumnSize - 1;
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
        this.dilation = dilation;
        this.isDepthSeparable = isDepthSeparable;
        this.asConvolution = !asConvolution && (filterRowSize > 1 && filterColumnSize > 1);
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
        inputGradient = outputGradient.getNewMatrix(inputGradientRowSize, inputGradientColumnSize, inputDepth);
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
                double gradientValue = filter.getValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth) * value;
                if (isDepthSeparable) {
                    inputGradient.addByValue(row + filterRow, column + filterColumn, depth, gradientValue);
                }
                else {
                    for (int currentInputDepth = 0; currentInputDepth < inputDepth; currentInputDepth++) {
                        inputGradient.addByValue(row + filterRow, column + filterColumn, currentInputDepth, gradientValue);
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
                if (!hasMaskAt(filterRow, filterColumn, depth, filter)) {
                    if (isDepthSeparable) {
                        inputGradient.addByValue(row + filterRow, column + filterColumn, depth, gradientValue);
                    }
                    else {
                        for (int currentInputDepth = 0; currentInputDepth < inputDepth; currentInputDepth++) {
                            inputGradient.addByValue(row + filterRow, column + filterColumn, currentInputDepth, gradientValue);
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
    private int getFilterRow(int filterRow) {
        return asConvolution ? filterRowSize - 1 - filterRow : filterRow;
    }

    /**
     * Returns filter column.
     *
     * @param filterColumn filter column.
     * @return filter column.
     */
    private int getFilterColumn(int filterColumn) {
        return asConvolution ? filterColumnSize - 1 - filterColumn : filterColumn;
    }

}
