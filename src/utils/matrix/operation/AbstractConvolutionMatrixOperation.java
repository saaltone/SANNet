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
public abstract class AbstractConvolutionMatrixOperation extends AbstractConvolutionOperation {

    /**
     * Total input depth.
     *
     */
    protected int totalInputDepth;

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
     * @param asConvolution    if true operation is executed as convolution otherwise as crosscorrelation
     */
    public AbstractConvolutionMatrixOperation(int rows, int columns, int depth, int filterRowSize, int filterColumnSize, int dilation, int stride, boolean isDepthSeparable, boolean asConvolution) {
        super(rows, columns, depth, filterRowSize, filterColumnSize, dilation, stride, isDepthSeparable, asConvolution);
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
        setTargetMatrix(input);
        setInputMatrix(filter);
        totalInputDepth = input.getDepth();
        setResult(input.getNewMatrix(getRows(), getColumns(), getDepth()));
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
                double filterValue = getInputMatrix().getValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth);
                double sumInputValue = 0;
                int inputRow = row + filterRow;
                int inputColumn = column + filterColumn;
                if (isDepthSeparable) {
                    sumInputValue = getTargetMatrix().getValue(inputRow, inputColumn, depth);
                }
                else {
                    for (int inputDepth = 0; inputDepth < totalInputDepth; inputDepth++) {
                        sumInputValue += getTargetMatrix().getValue(inputRow, inputColumn, inputDepth);
                    }
                }
                getResult().addByValue(row, column, depth, sumInputValue * filterValue);
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
                if (!hasMaskAt(filterRow, filterColumn, depth, getInputMatrix())) {
                    double filterValue = getInputMatrix().getValue(getFilterRow(filterRow), getFilterColumn(filterColumn), depth);
                    double sumInputValue = 0;
                    int inputRow = row + filterRow;
                    int inputColumn = column + filterColumn;
                    if (isDepthSeparable) {
                        if (!hasMaskAt(inputRow, inputColumn, depth, getTargetMatrix())) {
                            sumInputValue = getTargetMatrix().getValue(inputRow, inputColumn, depth);
                        }
                    }
                    else {
                        for (int inputDepth = 0; inputDepth < totalInputDepth; inputDepth++) {
                            if (!hasMaskAt(inputRow, inputColumn, inputDepth, getTargetMatrix())) {
                                sumInputValue += getTargetMatrix().getValue(inputRow, inputColumn, inputDepth);
                            }
                        }
                    }
                    getResult().addByValue(row, column, depth, sumInputValue * filterValue);
                }
            }
        }
    }

}
