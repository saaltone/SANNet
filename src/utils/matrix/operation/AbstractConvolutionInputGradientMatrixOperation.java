/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Defines abstract convolution input gradient matrix operation.
 *
 */
public abstract class AbstractConvolutionInputGradientMatrixOperation extends AbstractMatrixOperation {

    /**
     * Output gradient.
     *
     */
    protected Matrix outputGradient;

    /**
     * Filter matrix.
     *
     */
    protected Matrix filter;

    /**
     * Number of rows in filter.
     *
     */
    protected final int filterRowSize;

    /**
     * Number of columns in filter.
     *
     */
    protected final int filterColumnSize;

    /**
     * Resulting input gradient.
     *
     */
    protected Matrix inputGradient;

    /**
     * Dilation.
     *
     */
    protected final int dilation;

    /**
     * Constructor for abstract convolution input gradient matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param filterRowSize filter row size
     * @param filterColumnSize filter column size.
     * @param dilation dilation step
     * @param stride stride step
     */
    public AbstractConvolutionInputGradientMatrixOperation(int rows, int columns, int filterRowSize, int filterColumnSize, int dilation, int stride) {
        super(rows, columns, true, stride);
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
        this.dilation = dilation;
    }

    /**
     * Applies matrix operation.
     *
     * @param outputGradient output gradient.
     * @param filter filter matrix.
     * @param inputGradient input gradient.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix outputGradient, Matrix filter, Matrix inputGradient) throws MatrixException {
        this.outputGradient = outputGradient;
        this.filter = filter;
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

}
