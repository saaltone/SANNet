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
     * Input depth.
     *
     */
    protected int inputDepth;

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
        super(rows, columns, depth, filterRowSize, filterColumnSize, dilation, stride, isDepthSeparable, asConvolution, false);
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
        inputDepth = input.getDepth();
        setResult(input.getNewMatrix(getRows(), getColumns(), getDepth()));
        applyMatrixOperation();
        return getResult();
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
     */
    protected void applyOperation(int row, int column, int depth, int inputRow, int inputColumn, int filterRow, int filterColumn, double value) {
        double inputValue = 0;
        if (getIsDepthSeparable()) {
            inputValue = getTargetMatrix().getValue(inputRow, inputColumn, depth);
        }
        else {
            for (int inputDepth = 0; inputDepth < this.inputDepth; inputDepth++) {
                inputValue += getTargetMatrix().getValue(inputRow, inputColumn, inputDepth);
            }
        }
        double filterValue = getInputMatrix().getValue(filterRow, filterColumn, depth);
        double resultValue = inputValue * filterValue;
        getResult().addByValue(row, column, depth, resultValue);
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
     */
    protected void applyMaskOperation(int row, int column, int depth, int inputRow, int inputColumn, int filterRow, int filterColumn, double value) {
        double inputValue = 0;
        if (getIsDepthSeparable()) {
            if (!hasMaskAt(inputRow, inputColumn, depth, getTargetMatrix())) {
                inputValue = getTargetMatrix().getValue(inputRow, inputColumn, depth);
            }
        }
        else {
            for (int inputDepth = 0; inputDepth < this.inputDepth; inputDepth++) {
                if (!hasMaskAt(inputRow, inputColumn, inputDepth, getTargetMatrix())) {
                    inputValue += getTargetMatrix().getValue(inputRow, inputColumn, inputDepth);
                }
            }
        }
        double filterValue = getInputMatrix().getValue(filterRow, filterColumn, depth);
        double resultValue = inputValue * filterValue;
        getResult().addByValue(row, column, depth, resultValue);
    }

}
