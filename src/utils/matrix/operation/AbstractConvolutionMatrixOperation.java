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
     * Value of convolution.
     *
     */
    private transient double convolutionValue = 0;

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
        inputRows = input.getRows();
        inputColumns = input.getColumns();
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
        if (getIsDepthSeparable()) {
            double inputValue = getTargetMatrix().getValue(inputRow, inputColumn, depth);
            double filterValue = getInputMatrix().getValue(filterRow, filterColumn, depth);
            convolutionValue += inputValue * filterValue;
        }
        else {
            for (int inputDepth = 0; inputDepth < this.inputDepth; inputDepth++) {
                double inputValue = getTargetMatrix().getValue(inputRow, inputColumn, inputDepth);
                double filterValue = getInputMatrix().getValue(filterRow, filterColumn, getFilterPosition(inputDepth, depth));
                convolutionValue += inputValue * filterValue;
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
                double inputValue = getTargetMatrix().getValue(inputRow, inputColumn, depth);
                double filterValue = getInputMatrix().getValue(filterRow, filterColumn, depth);
                convolutionValue += inputValue * filterValue;
            }
        }
        else {
            for (int inputDepth = 0; inputDepth < this.inputDepth; inputDepth++) {
                if (!hasMaskAt(inputRow, inputColumn, inputDepth, getTargetMatrix())) {
                    double inputValue = getTargetMatrix().getValue(inputRow, inputColumn, inputDepth);
                    double filterValue = getInputMatrix().getValue(filterRow, filterColumn, getFilterPosition(inputDepth, depth));
                    convolutionValue += inputValue * filterValue;
                }
            }
        }
    }

    /**
     * Starts convolutional operation
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    protected void startOperation(int row, int column, int depth) {
        convolutionValue = 0;
    }

    /**
     * Finishes convolutional operation
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    protected void finishOperation(int row, int column, int depth) {
        getResult().setValue(row, column, depth, convolutionValue);
    }

    /**
     * Returns current input row.
     *
     * @param row row
     * @param filterRow filter row
     * @return current input row.
     */
    protected int getCurrentInputRow(int row, int filterRow) {
        return row + filterRow;
    }

    /**
     * Returns current input column.
     *
     * @param column column
     * @param filterColumn filter column
     * @return current input column.
     */
    protected int getCurrentInputColumn(int column, int filterColumn) {
        return column + filterColumn;
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
