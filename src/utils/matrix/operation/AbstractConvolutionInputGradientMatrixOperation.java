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
        super(rows, columns, depth, inputDepth, filterRowSize, filterColumnSize, dilation, stride, isDepthSeparable, !asConvolution, true);
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
        setResult(outputGradient.getNewMatrix(getInputRows(), getInputColumns(), getInputDepth()));
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
        if (getIsDepthSeparable()) {
            double filterValue = getInputMatrix().getValue(filterRow, filterColumn, depth);
            double gradientValue = filterValue * value;
            getResult().addByValue(inputRow, inputColumn, depth, gradientValue);
        }
        else {
            for (int inputDepth = 0; inputDepth < getInputDepth(); inputDepth++) {
                double filterValue = getInputMatrix().getValue(filterRow, filterColumn, getFilterPosition(inputDepth, depth));
                double gradientValue = filterValue * value;
                getResult().addByValue(inputRow, inputColumn, inputDepth, gradientValue);
            }
        }
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
        if (!hasMaskAt(filterRow, filterColumn, depth, getTargetMatrix())) {
            if (getIsDepthSeparable()) {
                double filterValue = getInputMatrix().getValue(filterRow, filterColumn, depth);
                double gradientValue = filterValue * value;
                getResult().addByValue(inputRow, inputColumn, depth, gradientValue);
            }
            else {
                for (int inputDepth = 0; inputDepth < getInputDepth(); inputDepth++) {
                    double filterValue = getInputMatrix().getValue(filterRow, filterColumn, getFilterPosition(inputDepth, depth));
                    double gradientValue = filterValue * value;
                    getResult().addByValue(inputRow, inputColumn, inputDepth, gradientValue);
                }
            }
        }
    }

}
