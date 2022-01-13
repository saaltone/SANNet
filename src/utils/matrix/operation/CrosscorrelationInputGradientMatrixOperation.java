/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.MatrixException;

/**
 * Defines crosscorrelation input gradient matrix operation.
 *
 */
public class CrosscorrelationInputGradientMatrixOperation extends AbstractConvolutionInputGradientMatrixOperation {

    /**
     * Constructor for crosscorrelation input gradient matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param filterRowSize filter row size
     * @param filterColumnSize filter column size.
     * @param dilation dilation step
     * @param stride stride step
     */
    public CrosscorrelationInputGradientMatrixOperation(int rows, int columns, int filterRowSize, int filterColumnSize, int dilation, int stride) {
        super(rows, columns, filterRowSize, filterColumnSize, dilation, stride);
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(int row, int column, double value) throws MatrixException {
        inputGradient.slice(row, column, row + filterRowSize - 1, column + filterColumnSize - 1);
        for (int filterRow = 0; filterRow < filterRowSize; filterRow += dilation) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn += dilation) {
                inputGradient.incrementByValue(filterRow, filterColumn, filter.getValue(filterRow, filterColumn) * value);
            }
        }
        inputGradient.unslice();
    }

    /**
     * Applies operation assuming masked matrices.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyMask(int row, int column, double value) throws MatrixException {
        inputGradient.slice(row, column, row + filterRowSize - 1, column + filterColumnSize - 1);
        for (int filterRow = 0; filterRow < filterRowSize; filterRow += dilation) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn += dilation) {
                if (!hasMaskAt(filterRow, filterColumn, filter)) {
                    inputGradient.incrementByValue(filterRow, filterColumn, filter.getValue(filterRow, filterColumn) * value);
                }
            }
        }
        inputGradient.unslice();
    }

}
