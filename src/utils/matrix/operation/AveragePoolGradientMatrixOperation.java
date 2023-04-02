/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements average pooling gradient matrix operation.
 *
 */
public class AveragePoolGradientMatrixOperation extends AbstractMatrixOperation {

    /**
     * Output gradient.
     *
     */
    private Matrix outputGradient;

    /**
     * Input gradient.
     *
     */
    private Matrix inputGradient;

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
     * Filter row size.
     *
     */
    private final int filterRowSize;

    /**
     * Filter column size.
     *
     */
    private final int filterColumnSize;

    /**
     * Inverted size of filter 1 / (filterRowSize * filterColumnSize)
     *
     */
    private final double invertedFilterSize;

    /**
     * Constructor for average pooling gradient matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param filterRowSize filter size in rows.
     * @param filterColumnSize filter size in columns.
     * @param stride stride step
     */
    public AveragePoolGradientMatrixOperation(int rows, int columns, int depth, int filterRowSize, int filterColumnSize, int stride) {
        super(rows, columns, depth, true, stride);
        this.inputGradientRowSize = rows + filterRowSize - 1;
        this.inputGradientColumnSize = columns + filterColumnSize - 1;
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
        this.invertedFilterSize = 1 / (double)(filterRowSize * filterColumnSize);
    }

    /**
     * Applies matrix operation.
     *
     * @param outputGradient output gradient.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix outputGradient) throws MatrixException {
        this.outputGradient = outputGradient;
        inputGradient = outputGradient.getNewMatrix(inputGradientRowSize, inputGradientColumnSize, getDepth());
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
        double gradientValue = value * invertedFilterSize;
        for (int filterRow = 0; filterRow < filterRowSize; filterRow++) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn++) {
                inputGradient.addByValue(row + filterRow, column + filterColumn, depth, gradientValue);
            }
        }
    }

}
