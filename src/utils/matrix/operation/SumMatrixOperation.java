/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.BinaryFunction;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;

/**
 * Implements sum matrix operation.
 *
 */
public class SumMatrixOperation extends AbstractMatrixOperation {

    /**
     * If value is one applies operation over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     *
     */
    private final int direction;

    /**
     * Cumulated value.
     *
     */
    private transient double value;

    /**
     * Number of counted entries.
     *
     */
    private transient int count;

    /**
     * Matrix for normalized values.
     *
     */
    private transient Matrix sumValues;

    /**
     * Multiply matrix operation.
     *
     */
    private final BinaryMatrixOperation multiplyMatrixOperation;

    /**
     * Constant matrix containing inverse number of rows.
     *
     */
    private final Matrix inverseRowsMatrix;

    /**
     * Constant matrix containing inverse number of columns.
     *
     */
    private final Matrix inverseColumnsMatrix;

    /**
     * Constant matrix containing inverse total depth.
     *
     */
    private final Matrix inverseDepthMatrix;

    /**
     * Constructor for sum matrix operation.
     *
     * @param rows          number of rows for operation.
     * @param columns       number of columns for operation.
     * @param depth         depth for operation.
     * @param direction     if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     */
    public SumMatrixOperation(int rows, int columns, int depth, int direction) {
        super(rows, columns, depth, true);
        this.direction = direction;
        multiplyMatrixOperation = new BinaryMatrixOperation(rows, columns, depth, new BinaryFunction((Matrix.MatrixBinaryOperation &Serializable) (value1, value2) -> value1 * value2));
        inverseRowsMatrix = new DMatrix(1 / (double)rows);
        inverseColumnsMatrix = new DMatrix(1 / (double)columns);
        inverseDepthMatrix = new DMatrix(1 / (double)depth);
    }

    /**
     * Applies sum operation.
     *
     * @param first first matrix.
     * @return sum of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double applySum(Matrix first) throws MatrixException {
        value = 0;
        count = 0;
        applyMatrixOperation(first, null, null);
        return value;
    }

    /**
     * Applies sum as matrix including normalization direction.
     *
     * @param first first matrix.
     * @return normalized value matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applySumAsMatrix(Matrix first) throws MatrixException {
        sumValues = first.getNewMatrix(getRows(), getColumns(), getDepth());
        switch(direction) {
            case 1, 2, 3 -> applyMatrixOperation(first, null, sumValues);
            default -> {
                sumValues = new DMatrix(applySum(first));
                return sumValues;
            }
        }
        if (!hasMask(first, null)) {
            for (int depth = 0; depth < getDepth(); depth++) {
                for (int row = 0; row < getRows(); row += getStride()) {
                    for (int column = 0; column < getColumns(); column += getStride()) {
                        switch(direction) {
                            case 1 -> sumValues.setValue(row, column, depth, sumValues.getValue(0, column, depth));
                            case 2 -> sumValues.setValue(row, column, depth, sumValues.getValue(row, 0, depth));
                            case 3 -> sumValues.setValue(row, column, depth, sumValues.getValue(row, column, 0));
                            default -> {
                            }
                        }
                    }
                }
            }
        }
        else {
            for (int depth = 0; depth < getDepth(); depth++) {
                for (int row = 0; row < getRows(); row += getStride()) {
                    for (int column = 0; column < getColumns(); column += getStride()) {
                        if (!hasMaskAt(row, column, depth, first, null)) {
                            switch(direction) {
                                case 1 -> sumValues.setValue(row, column, depth, sumValues.getValue(0, column, depth));
                                case 2 -> sumValues.setValue(row, column, depth, sumValues.getValue(row, 0, depth));
                                case 3 -> sumValues.setValue(row, column, depth, sumValues.getValue(row, column, 0));
                                default -> {
                                }
                            }
                        }
                    }
                }
            }
        }
        return sumValues;
    }

    /**
     * Applies mean operation.
     *
     * @param first first matrix.
     * @return mean of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double applyMean(Matrix first) throws MatrixException {
        return applySum(first) / (double)count;
    }

    /**
     * Applies mean as matrix including normalization direction.
     *
     * @param first first matrix.
     * @return normalized value matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyMeanAsMatrix(Matrix first) throws MatrixException {
        applySumAsMatrix(first);
        switch(direction) {
            case 1 -> {
                return multiplyMatrixOperation.applyFunction(sumValues, inverseRowsMatrix);
            }
            case 2 -> {
                return multiplyMatrixOperation.applyFunction(sumValues, inverseColumnsMatrix);
            }
            case 3 -> {
                return multiplyMatrixOperation.applyFunction(sumValues, inverseDepthMatrix);
            }
            default -> {
                return sumValues.divide(count);
            }
        }
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
        switch(direction) {
            case 1 -> result.setValue(0, column, depth, result.getValue(0, column, depth) + value);
            case 2 -> result.setValue(row, 0, depth, result.getValue(row, 0, depth) + value);
            case 3 -> result.setValue(row, column, 0, result.getValue(row, column, 0) + value);
            default -> {
                this.value += value;
                count++;
            }
        }
    }

}
