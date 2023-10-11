/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;

/**
 * Implements abstract positional pooling matrix operation.
 *
 */
public abstract class AbstractPositionalPoolingMatrixOperation extends AbstractConvolutionalOperation {

    /**
     * First matrix.
     *
     */
    private transient Matrix first;

    /**
     * Input position for each resulting row and column.
     *
     */
    private transient HashMap<Integer, Integer> inputPos;

    /**
     * Constructor for abstract positional pooling matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param filterRowSize filter size in rows.
     * @param filterColumnSize filter size in columns.
     * @param dilation dilation step
     * @param stride stride step
     */
    public AbstractPositionalPoolingMatrixOperation(int rows, int columns, int depth, int filterRowSize, int filterColumnSize, int dilation, int stride) {
        super(rows, columns, depth, depth, filterRowSize, filterColumnSize, dilation, stride, false);
    }

    /**
     * Applies matrix operation.
     *
     * @param first first matrix.
     * @param inputPos first positions.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix first, HashMap<Integer, Integer> inputPos) throws MatrixException {
        this.first = first;
        this.inputPos = inputPos;
        return applyMatrixOperation(first, null, first.getNewMatrix(getRows(), getColumns(), getDepth()));
    }

    /**
     * Returns first matrix.
     *
     * @return first matrix.
     */
    protected Matrix getFirst() {
        return first;
    }

    /**
     * Finishes convolutional operation
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    protected void finishOperation(int row, int column, int depth) {
        final int position = 2 * (depth * getRows() * getColumns() + column * getRows() + row);
        inputPos.put(position, getInputRow());
        inputPos.put(position + 1, getInputColumn());
    }

    /**
     * Returns input row.
     *
     * @return input row.
     */
    protected abstract int getInputRow();

    /**
     * Returns input column.
     *
     * @return input column.
     */
    protected abstract int getInputColumn();

}
