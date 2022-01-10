/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;

/**
 * Defines max pooling matrix operation.
 *
 */
public class MaxPoolMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input matrix.
     *
     */
    private Matrix input;

    /**
     * Number of inputs columns.
     *
     */
    private final int inputColumnSize;

    /**
     * Result.
     *
     */
    private Matrix result;

    /**
     * Number of rows in filter.
     *
     */
    private final int filterRowSize;

    /**
     * Number of columns in filter.
     *
     */
    private final int filterColumnSize;

    /**
     * Maximum position for each resulting row and column.
     *
     */
    private HashMap<Integer, Integer> maxPos;

    /**
     * Constructor for max pooling matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param inputColumnSize number of input columns.
     * @param filterRowSize filter size in rows.
     * @param filterColumnSize filter size in columns.
     * @param stride stride step
     */
    public MaxPoolMatrixOperation(int rows, int columns, int inputColumnSize, int filterRowSize, int filterColumnSize, int stride) {
        super(rows, columns, false, stride);
        this.inputColumnSize = inputColumnSize;
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
    }

    /**
     * Applies matrix operation.
     *
     * @param input input matrix.
     * @param maxPos maximum positions.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(Matrix input, HashMap<Integer, Integer> maxPos, Matrix result) throws MatrixException {
        this.input = input;
        this.maxPos = maxPos;
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

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(int row, int column, double value) throws MatrixException {
        input.sliceAt(row, column, row + filterRowSize - 1, column + filterColumnSize - 1);
        int maxRow = -1;
        int maxColumn = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int filterRow = 0; filterRow < filterRowSize; filterRow++) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn++) {
                double filterValue = input.getValue(filterRow, filterColumn);
                if (maxValue < filterValue) {
                    maxValue = filterValue;
                    maxRow = filterRow;
                    maxColumn = filterColumn;
                }
            }
        }
        result.setValue(row, column, maxValue);
        maxPos.put(2 * (row * inputColumnSize + column), row + maxRow);
        maxPos.put(2 * (row * inputColumnSize + column) + 1, column + maxColumn);
        input.unslice();
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
        input.sliceAt(row, column, row + filterRowSize - 1, column + filterColumnSize - 1);
        int maxRow = -1;
        int maxColumn = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int filterRow = 0; filterRow < filterRowSize; filterRow++) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn++) {
                if (!hasMaskAt(filterRow, filterColumn, input)) {
                    int inputRow = row + filterRow;
                    int inputColumn = column + filterColumn;
                    double inputValue = input.getValue(inputRow, inputColumn);
                    if (maxValue < inputValue) {
                        maxValue = inputValue;
                        maxRow = inputRow;
                        maxColumn = inputColumn;
                    }
                }
            }
        }
        result.setValue(row, column, maxValue);
        maxPos.put(2 * (row * inputColumnSize + column), maxRow);
        maxPos.put(2 * (row * inputColumnSize + column) + 1, maxColumn);
        input.unslice();
    }

}
