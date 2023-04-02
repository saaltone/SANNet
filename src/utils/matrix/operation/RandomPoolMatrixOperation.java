/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import java.util.ArrayList;
import java.util.Random;

/**
 * Implements random pooling matrix operation.<br>
 * Selects each input of pool for propagation randomly with uniform probability.<br>
 *
 */
public class RandomPoolMatrixOperation extends AbstractPositionalPoolingMatrixOperation {

    /**
     * Current input row.
     *
     */
    private int inputRow;

    /**
     * Current input column.
     *
     */
    private int inputColumn;

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
     * @param depth depth for operation.
     * @param inputRowSize number of input rows.
     * @param inputColumnSize number of input columns.
     * @param filterRowSize filter size in rows.
     * @param filterColumnSize filter size in columns.
     * @param stride stride step
     */
    public RandomPoolMatrixOperation(int rows, int columns, int depth, int inputRowSize, int inputColumnSize, int filterRowSize, int filterColumnSize, int stride) {
        super(rows, columns, depth, filterRowSize, filterColumnSize, stride);
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    public void executeApply(int row, int column, int depth) {
        inputRow = row + random.nextInt(filterRowSize);
        inputColumn = column + random.nextInt(filterRowSize);
        result.setValue(row, column, depth, input.getValue(inputRow, inputColumn, depth));
    }

    /**
     * Applies operation assuming masked matrices.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     */
    public void executeApplyMask(int row, int column, int depth) {
        ArrayList<Integer> availableRows = new ArrayList<>();
        ArrayList<Integer> availableColumns = new ArrayList<>();
        for (int filterRow = 0; filterRow < filterRowSize; filterRow++) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn++) {
                int inputRow = row + filterRow;
                int inputColumn = column + filterColumn;
                if (!hasMaskAt(inputRow, inputColumn, depth, input)) {
                    availableRows.add(inputRow);
                    availableColumns.add(inputColumn);
                }
            }
        }
        int pos = random.nextInt(availableRows.size());
        inputRow = availableRows.get(pos);
        inputColumn = availableColumns.get(pos);
        result.setValue(row, column, depth, input.getValue(inputRow, inputColumn, depth));
    }

    /**
     * Returns input row.
     *
     * @return input row.
     */
    protected int getInputRow() {
        return inputRow;
    }

    /**
     * Returns input column.
     *
     * @return input column.
     */
    protected int getInputColumn() {
        return inputColumn;
    }

}
