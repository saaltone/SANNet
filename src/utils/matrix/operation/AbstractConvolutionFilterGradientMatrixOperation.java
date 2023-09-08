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
     * Input gradient row size.
     *
     */
    private int inputRows;

    /**
     * Input gradient column size.
     *
     */
    private int inputColumns;

    /**
     * Input depth.
     *
     */
    protected int inputDepth;

    /**
     * Half filter rows.
     *
     */
    private final int halfFilterRows;

    /**
     * Half filter columns.
     *
     */
    private final int halfFilterColumns;

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
        halfFilterRows = filterRowSize / 2;
        halfFilterColumns = filterColumnSize / 2;
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
        inputRows = outputGradient.getRows() + getFilterRows() - 1;
        inputColumns = outputGradient.getColumns() + getFilterColumns() - 1;
        inputDepth = input.getDepth();
        setResult(outputGradient.getNewMatrix(getFilterRows(), getFilterColumns(), getIsDepthSeparable() ? inputDepth : inputDepth * getDepth()));
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
            double inputValue = getInputMatrix().getValue(inputRow, inputColumn, depth);
            getResult().addByValue(filterRow, filterColumn, depth, inputValue * value);
        }
        else {
            for (int inputDepth = 0; inputDepth < this.inputDepth; inputDepth++) {
                double inputValue = getInputMatrix().getValue(inputRow, inputColumn, inputDepth);
                getResult().addByValue(filterRow, filterColumn, getFilterPosition(inputDepth, depth), inputValue * value);
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
        if (getIsDepthSeparable()) {
            if (!hasMaskAt(inputRow, inputColumn, depth, getTargetMatrix())) {
                double inputValue = getInputMatrix().getValue(inputRow, inputColumn, depth);
                getResult().addByValue(filterRow, filterColumn, depth, inputValue * value);
            }
        }
        else {
            for (int inputDepth = 0; inputDepth < this.inputDepth; inputDepth++) {
                if (!hasMaskAt(inputRow, inputColumn, inputDepth, getTargetMatrix())) {
                    double inputValue = getInputMatrix().getValue(inputRow, inputColumn, depth);
                    getResult().addByValue(filterRow, filterColumn, getFilterPosition(inputDepth, depth), inputValue * value);
                }
            }
        }
    }

    /**
     * Returns current input row.
     *
     * @param row row
     * @param filterRow filter row
     * @return current input row.
     */
    protected int getCurrentInputRow(int row, int filterRow) {
        return row + filterRow - halfFilterRows;
    }

    /**
     * Returns current input column.
     *
     * @param column column
     * @param filterColumn filter column
     * @return current input column.
     */
    protected int getCurrentInputColumn(int column, int filterColumn) {
        return column + filterColumn - halfFilterColumns;
    }

    /**
     * Checks if input row and columns are valid.
     *
     * @param inputRow input row
     * @param inputColumn input column
     * @return true if input row and column are valid otherwise returns false.
     */
    protected boolean isValidInputPosition(int inputRow, int inputColumn) {
        return (inputRow >= 0 && inputColumn >= 0 && inputRow < inputRows && inputColumn < inputColumns);
    }

}
