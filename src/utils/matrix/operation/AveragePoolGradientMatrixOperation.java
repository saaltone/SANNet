/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Defines average pooling gradient matrix operation.
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
     * Inverted size of filter 1 / (filterRowSize * filterColumnSize)
     *
     */
    private final double invertedFilterSize;

    /**
     * Constructor for average pooling gradient matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param filterRowSize filter size in rows.
     * @param filterColumnSize filter size in columns.
     * @param stride stride step
     */
    public AveragePoolGradientMatrixOperation(int rows, int columns, int filterRowSize, int filterColumnSize, int stride) {
        super(rows, columns, false, stride);
        this.invertedFilterSize = 1 / (double)(filterRowSize * filterColumnSize);
    }

    /**
     * Applies matrix operation.
     *
     * @param outputGradient output gradient.
     * @param inputGradient input gradient.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix outputGradient, Matrix inputGradient) throws MatrixException {
        this.outputGradient = outputGradient;
        this.inputGradient = inputGradient;
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
    public Matrix getAnother() {
        return null;
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     */
    public void apply(int row, int column, double value) {
        inputGradient.setValue(row, column, value * invertedFilterSize);
    }

}
