/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.Random;

/**
 * Implements weight random sampling (choice) matrix operation. Assumes that matrix is distribution summing to 1 with values between 0 and 1.
 *
 */
public class SampleMatrixOperation extends AbstractMatrixOperation {

    /**
     * Cumulative value for sampling.
     *
     */
    private double cumulativeValue = 0;

    /**
     * Threshold value for sampling.
     *
     */
    private double thresholdValue = Double.MIN_VALUE;

    /**
     * Selected row.
     *
     */
    private transient int selectedRow = -1;

    /**
     * Selected column.
     *
     */
    private transient int selectedColumn = -1;

    /**
     * Selected depth.
     *
     */
    private transient int selectedDepth = -1;

    /**
     * Random function.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for sample matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     */
    public SampleMatrixOperation(int rows, int columns, int depth) {
        super(rows, columns, depth, true);
    }

    /**
     * Applies sample operation.
     *
     * @param first first matrix.
     * @return maximum arguments (row and column)
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public int[] sample(Matrix first) throws MatrixException {
        cumulativeValue = 0;
        thresholdValue = random.nextDouble();
        selectedRow = -1;
        selectedColumn = -1;
        selectedDepth = -1;
        applyMatrixOperation(first, null, null);
        int[] result = new int[3];
        result[0] = getSelectedRow();
        result[1] = getSelectedColumn();
        result[2] = getSelectedDepth();
        return result;
    }

    /**
     * Applies operation.
     *
     * @param row    current row.
     * @param column current column.
     * @param depth  current depth.
     * @param value  current value.
     * @param result result matrix.
     */
    public void apply(int row, int column, int depth, double value, Matrix result) {
        cumulativeValue += value;
        if (cumulativeValue >= thresholdValue && selectedRow == -1 && selectedColumn == -1 && selectedDepth == -1) {
            selectedRow = row;
            selectedColumn = column;
            selectedDepth = depth;
        }
    }

    /**
     * Returns selected row.
     *
     * @return selected row.
     */
    public int getSelectedRow() {
        return selectedRow;
    }

    /**
     * Returns selected column.
     *
     * @return selected column.
     */
    public int getSelectedColumn() {
        return selectedColumn;
    }

    /**
     * Returns selected depth.
     *
     * @return selected depth.
     */
    public int getSelectedDepth() {
        return selectedDepth;
    }

}
