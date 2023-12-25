/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.configurable.DynamicParamException;
import utils.matrix.*;

/**
 * Implements variance matrix operation.
 *
 */
public class VarianceMatrixOperation extends AbstractMatrixOperation {

    /**
     * If value is one applies operation over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     *
     */
    private final int direction;

    /**
     * Mean value for variance operation.
     *
     */
    private final double mean;

    /**
     * Mean value matrix for variance operation.
     *
     */
    private Matrix meanMatrix;

    /**
     * Cumulated variance value.
     *
     */
    private transient double value;

    /**
     * Number of counted entries.
     *
     */
    private transient int count;

    /**
     * Square root matrix operation.
     *
     */
    private final UnaryMatrixOperation sqrtMatrixOperation;

    /**
     * Constructor for variance operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param mean mean value for variance operation.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public VarianceMatrixOperation(int rows, int columns, int depth, double mean) throws MatrixException, DynamicParamException {
        super(rows, columns, depth, true);
        this.mean = mean;
        this.meanMatrix = null;
        this.direction = 0;
        sqrtMatrixOperation = new UnaryMatrixOperation(rows, columns, depth, new UnaryFunction(UnaryFunctionType.SQRT));
    }

    /**
     * Constructor for variance operation.
     *
     * @param rows       number of rows for operation.
     * @param columns    number of columns for operation.
     * @param depth      depth for operation.
     * @param direction  if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public VarianceMatrixOperation(int rows, int columns, int depth, int direction) throws MatrixException, DynamicParamException {
        super(rows, columns, depth, true);
        this.mean = 0;
        this.direction = direction;
        sqrtMatrixOperation = new UnaryMatrixOperation(rows, columns, depth, new UnaryFunction(UnaryFunctionType.SQRT));
    }

    /**
     * Applies variance operation.
     *
     * @param input input matrix.
     * @return variance of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double applyVariance(Matrix input) throws MatrixException {
        value = 0;
        count = 0;
        applyMatrixOperation(input, null, null);
        return count > 0 ? value / (double)count : 0;
    }

    /**
     * Applies variance as matrix including normalization direction.
     *
     * @param first first matrix.
     * @param meanMatrix mean value matrix for variance operation.
     * @return normalized value matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyVarianceAsMatrix(Matrix first, Matrix meanMatrix) throws MatrixException {
        this.meanMatrix = meanMatrix;
        Matrix varianceValues = first.getNewMatrix(getRows(), getColumns(), getDepth());
        switch(direction) {
            case 1, 2, 3 -> applyMatrixOperation(first, null, varianceValues);
            default -> {
                varianceValues = new DMatrix(applyVariance(first));
                return varianceValues.divide(count);
            }
        }
        int rows = getRows();
        int columns = getColumns();
        int totalDepth = getDepth();
        if (!hasMask(first, null)) {
            for (int depth = 0; depth < totalDepth; depth++) {
                for (int row = 0; row < rows; row += getStride()) {
                    for (int column = 0; column < columns; column += getStride()) {
                        switch(direction) {
                            case 1 -> varianceValues.setValue(row, column, depth, varianceValues.getValue(0, column, depth) / rows);
                            case 2 -> varianceValues.setValue(row, column, depth, varianceValues.getValue(row, 0, depth) / columns);
                            case 3 -> varianceValues.setValue(row, column, depth, varianceValues.getValue(row, column, 0) / totalDepth);
                            default -> {
                            }
                        }
                    }
                }
            }
        }
        else {
            for (int depth = 0; depth < totalDepth; depth++) {
                for (int row = 0; row < rows; row += getStride()) {
                    for (int column = 0; column < columns; column += getStride()) {
                        if (!hasMaskAt(row, column, depth, first, null)) {
                            switch(direction) {
                                case 1 -> varianceValues.setValue(row, column, depth, varianceValues.getValue(0, column, depth) / rows);
                                case 2 -> varianceValues.setValue(row, column, depth, varianceValues.getValue(row, 0, depth) / columns);
                                case 3 -> varianceValues.setValue(row, column, depth, varianceValues.getValue(row, column, 0) / totalDepth);
                                default -> {
                                }
                            }
                        }
                    }
                }
            }
        }
        return varianceValues;
    }

    /**
     * Applies standard deviation operation.
     *
     * @param input input matrix.
     * @return standard deviation of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double applyStandardDeviation(Matrix input) throws MatrixException {
        value = 0;
        count = 0;
        applyMatrixOperation(input, null, null);
        return count > 0 ? Math.sqrt(value / (double)(count)) : 0;
    }

    /**
     * Applies mean as matrix including normalization direction.
     *
     * @param first first matrix.
     * @param meanMatrix mean value matrix for variance operation.
     * @return normalized value matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyStandardDeviationAsMatrix(Matrix first, Matrix meanMatrix) throws MatrixException {
        return sqrtMatrixOperation.applyFunction(applyVarianceAsMatrix(first, meanMatrix));
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
            case 1 -> result.setValue(0, column, depth, result.getValue(0, column, depth) + Math.pow(value - meanMatrix.getValue(row, column, depth), 2));
            case 2 -> result.setValue(row, 0, depth, result.getValue(row, 0, depth) + Math.pow(value - meanMatrix.getValue(row, column, depth), 2));
            case 3 -> result.setValue(row, column, 0, result.getValue(row, column, 0) + Math.pow(value - meanMatrix.getValue(row, column, depth), 2));
            default -> {
                this.value += Math.pow(value - mean, 2);
                count++;
            }
        }
        this.value += Math.pow(value - mean, 2);
        count++;
    }

}
