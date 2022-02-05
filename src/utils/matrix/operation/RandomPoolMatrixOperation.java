/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/**
 * Implements random pooling matrix operation.<br>
 * Selects each input of pool for propagation randomly with uniform probability.<br>
 *
 */
public class RandomPoolMatrixOperation extends AbstractMatrixOperation {

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
     * Input position for each resulting row and column.
     *
     */
    private HashMap<Integer, Integer> inputPos;

    /**
     * Random number generator.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for random pooling matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param inputColumnSize number of input columns.
     * @param filterRowSize filter size in rows.
     * @param filterColumnSize filter size in columns.
     * @param stride stride step
     */
    public RandomPoolMatrixOperation(int rows, int columns, int inputColumnSize, int filterRowSize, int filterColumnSize, int stride) {
        super(rows, columns, false, stride);
        this.inputColumnSize = inputColumnSize;
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
    }

    /**
     * Applies matrix operation.
     *
     * @param input input matrix.
     * @param inputPos input positions.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(Matrix input, HashMap<Integer, Integer> inputPos, Matrix result) throws MatrixException {
        this.input = input;
        this.inputPos = inputPos;
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
        input.slice(row, column, row + filterRowSize - 1, column + filterColumnSize - 1);
        int filterRow = random.nextInt(filterRowSize);
        int filterColumn = random.nextInt(filterRowSize);
        double inputValue = input.getValue(filterRow, filterColumn);
        result.setValue(row, column, inputValue);
        inputPos.put(2 * (row * inputColumnSize + column), row + filterRow);
        inputPos.put(2 * (row * inputColumnSize + column) + 1, column + filterColumn);
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
        input.slice(row, column, row + filterRowSize - 1, column + filterColumnSize - 1);
        ArrayList<Integer> availableRows = new ArrayList<>();
        ArrayList<Integer> availableColumns = new ArrayList<>();
        for (int filterRow = 0; filterRow < filterRowSize; filterRow++) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn++) {
                if (!hasMaskAt(filterRow, filterColumn, input)) {
                    availableRows.add(filterRow);
                    availableColumns.add(filterColumn);
                }
            }
        }
        int pos = random.nextInt(availableRows.size());
        int filterRow = availableRows.get(pos);
        int filterColumn = availableColumns.get(pos);
        double inputValue = input.getValue(filterRow, filterColumn);
        result.setValue(row, column, inputValue);
        inputPos.put(2 * (row * inputColumnSize + column), row + filterRow);
        inputPos.put(2 * (row * inputColumnSize + column) + 1, column + filterColumn);
        input.unslice();
    }

}
