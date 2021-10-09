/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Defines abstract convolution matrix operation.
 *
 */
public abstract class AbstractConvolutionMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input matrix.
     *
     */
    protected Matrix input;

    /**
     * Filter matrix.
     *
     */
    protected Matrix filter;

    /**
     * Result matrix.
     *
     */
    protected Matrix result;

    /**
     * Matrix dilation value.
     *
     */
    protected final int dilation;

    /**
     * Filter row size.
     *
     */
    protected final int filterRowSize;

    /**
     * Filter column size.
     *
     */
    protected final int filterColumnSize;

    /**
     * Constructor for abstract convolution matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param filterRowSize filter row size
     * @param filterColumnSize filter column size.
     * @param dilation dilation step
     * @param stride stride step
     */
    public AbstractConvolutionMatrixOperation(int rows, int columns, int filterRowSize, int filterColumnSize, int dilation, int stride) {
        super(rows, columns, false, stride);
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
        this.dilation = dilation;
    }

    /**
     * Applies matrix operation.
     *
     * @param input input matrix.
     * @param filter filter matrix.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(Matrix input, Matrix filter, Matrix result) throws MatrixException {
        this.input = input;
        this.filter = filter;
        this.result = result;
        applyMatrixOperation();
    }

    /**
     * Returns target matrix.
     *
     * @return target matrix.
     */
    protected Matrix getTargetMatrix() {
        return input;
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
