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
public abstract class AbstractPositionalPoolingMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input matrix.
     *
     */
    protected Matrix input;

    /**
     * Result.
     *
     */
    protected Matrix result;

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
     * Input position for each resulting row and column.
     *
     */
    private HashMap<Integer, Integer> inputPos;

    /**
     * Constructor for abstract positional pooling matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param filterRowSize filter size in rows.
     * @param filterColumnSize filter size in columns.
     * @param stride stride step
     */
    public AbstractPositionalPoolingMatrixOperation(int rows, int columns, int depth, int filterRowSize, int filterColumnSize, int stride) {
        super(rows, columns, depth, false, stride);
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
    }

    /**
     * Applies matrix operation.
     *
     * @param input input matrix.
     * @param inputPos input positions.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix input, HashMap<Integer, Integer> inputPos) throws MatrixException {
        this.input = input;
        this.inputPos = inputPos;
        this.result = input.getNewMatrix(getRows(), getColumns(), getDepth());
        applyMatrixOperation();
        return result;
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
        executeApply(row, column, depth);
        updateInputPosition(row, column, depth);
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    protected abstract void executeApply(int row, int column, int depth);

    /**
     * Applies operation assuming masked matrices.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     * @param value current value.
     */
    public void applyMask(int row, int column, int depth, double value) {
        executeApplyMask(row, column, depth);
        updateInputPosition(row, column, depth);
    }

    /**
     * Applies operation assuming masked matrices.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    protected abstract void executeApplyMask(int row, int column, int depth);

    /**
     * Updates input position.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    private void updateInputPosition(int row, int column, int depth) {
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
