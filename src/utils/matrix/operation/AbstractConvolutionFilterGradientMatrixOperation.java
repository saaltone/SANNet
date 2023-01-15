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
public abstract class AbstractConvolutionFilterGradientMatrixOperation extends AbstractMatrixOperation {

    /**
     * Output gradient.
     *
     */
    protected Matrix outputGradient;

    /**
     * Input matrix.
     *
     */
    protected Matrix input;

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
     * Dilation.
     *
     */
    protected final int dilation;

    /**
     * Resulting filter gradient.
     *
     */
    protected Matrix filterGradient;

    /**
     * Constructor for abstract convolution filter gradient matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param filterRowSize filter row size
     * @param filterColumnSize filter column size.
     * @param dilation dilation step
     * @param stride stride step
     */
    public AbstractConvolutionFilterGradientMatrixOperation(int rows, int columns, int filterRowSize, int filterColumnSize, int dilation, int stride) {
        super(rows, columns, true, stride);
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
        this.dilation = dilation;
    }

    /**
     * Applies matrix operation.
     *
     * @param outputGradient output gradient.
     * @param input input matrix.
     * @param filterGradient filter gradient.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix outputGradient, Matrix input, Matrix filterGradient) throws MatrixException {
        this.outputGradient = outputGradient;
        this.input = input;
        this.filterGradient = filterGradient;
        applyMatrixOperation();
        return filterGradient;
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
