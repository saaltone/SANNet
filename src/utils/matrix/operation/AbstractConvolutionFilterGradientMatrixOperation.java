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
public abstract class AbstractConvolutionFilterGradientMatrixOperation extends AbstractConvolutionOperation {

    /**
     * Total input depth.
     *
     */
    protected int totalInputDepth;

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
     * @param asConvolution    if true operation is executed as convolution otherwise as crosscorrelation
     */
    public AbstractConvolutionFilterGradientMatrixOperation(int rows, int columns, int depth, int filterRowSize, int filterColumnSize, int dilation, int stride, boolean isDepthSeparable, boolean asConvolution) {
        super(rows, columns, depth, filterRowSize, filterColumnSize, dilation, stride, isDepthSeparable, asConvolution);
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
        setTargetMatrix(outputGradient);
        setInputMatrix(input);
        totalInputDepth = input.getDepth();
        setResult(outputGradient.getNewMatrix(filterRowSize, filterColumnSize, getDepth()));
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
                int inputRow = row + filterRow;
                int inputColumn = column + filterColumn;
                if (isDepthSeparable) {
                    double inputValue = getInputMatrix().getValue(inputRow, inputColumn, depth);
                    getResult().addByValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth, inputValue * value);
                }
                else {
                    for (int inputDepth = 0; inputDepth < totalInputDepth; inputDepth++) {
                        double inputValue = getInputMatrix().getValue(inputRow, inputColumn, inputDepth);
                        getResult().addByValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth, inputValue * value);
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
                int inputRow = row + filterRow;
                int inputColumn = column + filterColumn;
                if (isDepthSeparable) {
                    if (!hasMaskAt(inputRow, inputColumn, depth, getTargetMatrix())) {
                        double inputValue = getInputMatrix().getValue(inputRow, inputColumn, depth);
                        getResult().addByValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth, inputValue * value);
                    }
                }
                else {
                    for (int inputDepth = 0; inputDepth < totalInputDepth; inputDepth++) {
                        if (!hasMaskAt(inputRow, column + filterColumn, inputDepth, getTargetMatrix())) {
                            double inputValue = getInputMatrix().getValue(inputRow, column + filterColumn, inputDepth);
                            getResult().addByValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth, inputValue * value);
                        }
                    }
                }
            }
        }
    }

}
