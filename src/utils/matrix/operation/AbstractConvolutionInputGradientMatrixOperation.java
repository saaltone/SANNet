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
public abstract class AbstractConvolutionInputGradientMatrixOperation extends AbstractConvolutionOperation {

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
     * Input depth.
     *
     */
    protected final int inputDepth;

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
        super(rows, columns, depth, filterRowSize, filterColumnSize, dilation, stride, isDepthSeparable, !asConvolution);
        this.inputDepth = inputDepth;
        this.inputGradientRowSize = rows + filterRowSize - 1;
        this.inputGradientColumnSize = columns + filterColumnSize - 1;
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
        setTargetMatrix(outputGradient);
        setInputMatrix(filter);
        setResult(outputGradient.getNewMatrix(inputGradientRowSize, inputGradientColumnSize, inputDepth));
        applyMatrixOperation();
        return getResult();
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
                double gradientValue = getInputMatrix().getValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth) * value;
                int inputRow = row + filterRow;
                int inputColumn = column + filterColumn;
                if (isDepthSeparable) {
                    getResult().addByValue(inputRow, inputColumn, depth, gradientValue);
                }
                else {
                    for (int inputDepth = 0; inputDepth < this.inputDepth; inputDepth++) {
                        getResult().addByValue(inputRow, inputColumn, inputDepth, gradientValue);
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
                double gradientValue = getInputMatrix().getValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth) * value;
                int inputRow = row + filterRow;
                int inputColumn = column + filterColumn;
                if (!hasMaskAt(filterRow, filterColumn, depth, getTargetMatrix())) {
                    if (isDepthSeparable) {
                        getResult().addByValue(inputRow, inputColumn, depth, gradientValue);
                    }
                    else {
                        for (int inputDepth = 0; inputDepth < this.inputDepth; inputDepth++) {
                            getResult().addByValue(inputRow, inputColumn, inputDepth, gradientValue);
                        }
                    }
                }
            }
        }
    }

}
