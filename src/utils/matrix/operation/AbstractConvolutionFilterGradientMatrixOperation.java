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
     * Input depth.
     *
     */
    protected int inputDepth;

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
        super(rows, columns, depth, filterRowSize, filterColumnSize, dilation, stride, isDepthSeparable, asConvolution, true);
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
        inputDepth = input.getDepth();
        setResult(outputGradient.getNewMatrix(getFilterRows(), getFilterColumns(), getDepth()));
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
            inputValue = getInputMatrix().getValue(inputRow, inputColumn, depth);
        }
        else {
            for (int inputDepth = 0; inputDepth < this.inputDepth; inputDepth++) {
                inputValue += getInputMatrix().getValue(inputRow, inputColumn, inputDepth);
            }
        }
        double gradientValue = inputValue * value;
        getResult().addByValue(filterRow, filterColumn, depth, gradientValue);
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
                inputValue = getInputMatrix().getValue(inputRow, inputColumn, depth);
            }
        }
        else {
            for (int inputDepth = 0; inputDepth < this.inputDepth; inputDepth++) {
                if (!hasMaskAt(inputRow, inputColumn, inputDepth, getTargetMatrix())) {
                    inputValue += getInputMatrix().getValue(inputRow, column + filterColumn, inputDepth);
                }
            }
        }
        double gradientValue = inputValue * value;
        getResult().addByValue(filterRow, filterColumn, depth, gradientValue);
    }

}
