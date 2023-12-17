/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.Random;

/**
 * Implements softmax matrix operation.
 *
 */
public class SoftmaxMatrixOperation extends AbstractMatrixOperation {

    /**
     * Tau value for softmax.
     *
     */
    private final double tau;

    /**
     * If true applies Gumbel softmax otherwise normal softmax.
     *
     */
    private final boolean gumbelSoftmax;

    /**
     * Random function.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for softmax matrix operation.
     *
     * @param rows          number of rows for operation.
     * @param columns       number of columns for operation.
     * @param depth         depth for operation.
     * @param tau           tau (temperature) value
     * @param gumbelSoftmax if true applies gumbel softmax other normal softmax.
     */
    public SoftmaxMatrixOperation(int rows, int columns, int depth, double tau, boolean gumbelSoftmax) {
        super(rows, columns, depth, true);
        this.tau = tau;
        this.gumbelSoftmax = gumbelSoftmax;
    }

    /**
     * Applies operation.
     *
     * @param first first matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyFunction(Matrix first) throws MatrixException {
        final int rows = getRows();
        final int columns = getColumns();
        final int totalDepth = getDepth();
        final double[] rowValues = new double[rows];
        final Matrix result = first.getNewMatrix(rows, columns, totalDepth);
        if (!hasMask(first, null)) {
            for (int depth = 0; depth < totalDepth; depth++) {
                for (int column = 0; column < columns; column += getStride()) {
                    double maxValue = 0;
                    for (int row = 0; row < rows; row += getStride()) {
                        double value = gumbelSoftmax ? (first.getValue(row, column, depth) + getGumbelNoise()) / tau : first.getValue(row, column, depth)/ tau;
                        rowValues[row] = value;
                        maxValue = row == 0 ? value : Math.max(maxValue, value);
                    }
                    double rowSum = 0;
                    for (int row = 0; row < rows; row += getStride()) {
                        double rowExpValue = Math.exp(rowValues[row] - maxValue);
                        rowValues[row] = rowExpValue;
                        rowSum += rowExpValue;
                    }
                    for (int row = 0; row < rows; row += getStride()) {
                        result.setValue(row, column, depth, rowValues[row] / rowSum);
                    }
                }
            }
        }
        else {
            for (int depth = 0; depth < totalDepth; depth++) {
                for (int column = 0; column < columns; column += getStride()) {
                    double maxValue = 0;
                    for (int row = 0; row < rows; row += getStride()) {
                        if (!hasMaskAt(row, column, depth, first, null)) {
                            double value = gumbelSoftmax ? (first.getValue(row, column, depth) + getGumbelNoise()) / tau : first.getValue(row, column, depth)/ tau;
                            rowValues[row] = value;
                            maxValue = row == 0 ? value : Math.max(maxValue, value);
                        }
                    }
                    double rowSum = 0;
                    for (int row = 0; row < rows; row += getStride()) {
                        if (!hasMaskAt(row, column, depth, first, null)) {
                            double rowExpValue = Math.exp(rowValues[row] - maxValue);
                            rowValues[row] = rowExpValue;
                            rowSum += rowExpValue;
                        }
                    }
                    for (int row = 0; row < rows; row += getStride()) {
                        if (!hasMaskAt(row, column, depth, first, null)) {
                            result.setValue(row, column, depth, rowValues[row] / rowSum);
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * Applies gradient operation.
     *
     * @param first first matrix.
     * @param outputGradient output gradient.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyGradient(Matrix first, Matrix outputGradient) throws MatrixException {
        final int rows = getRows();
        final int columns = getColumns();
        final int totalDepth = getDepth();
        final Matrix gradient = first.getNewMatrix(rows, columns, totalDepth);
        if (!hasMask(first, null)) {
            for (int depth = 0; depth < totalDepth; depth++) {
                for (int column = 0; column < columns; column += getStride()) {
                    for (int row0 = 0; row0 < rows; row0 += getStride()) {
                        for (int row = 0; row < rows; row += getStride()) {
                            double firstValue = first.getValue(row, column, depth);
                            double secondValue = first.getValue(row0, column, depth);
                            double gradientValue = firstValue * ((row == row0 ? 1 : 0) - secondValue);
                            gradient.setValue(row0, column, depth, gradient.getValue(row0, column, depth) + gradientValue * outputGradient.getValue(row, column, depth));
                        }
                    }
                }
            }
        }
        else {
            for (int depth = 0; depth < getDepth(); depth++) {
                for (int column = 0; column < columns; column += getStride()) {
                    for (int row0 = 0; row0 < getRows(); row0 += getStride()) {
                        for (int row = 0; row < getRows(); row += getStride()) {
                            if (!hasMaskAt(row, 0, depth, first, null)) {
                                double firstValue = first.getValue(row, column, depth);
                                double secondValue = first.getValue(row0, column, depth);
                                double gradientValue = firstValue * ((row == row0 ? 1 : 0) - secondValue);
                                gradient.setValue(row0, column, depth, gradient.getValue(row0, column, depth) + gradientValue * outputGradient.getValue(row, column, depth));
                            }
                        }
                    }
                }
            }
        }
        return gradient;
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
    }


    /**
     * Returns Gumbel noise.<br>
     *
     * @return Gumbel noise.
     */
    private double getGumbelNoise() {
        double epsilon = 10E-20;
        return -Math.log(-Math.log(random.nextDouble() + epsilon) + epsilon);
    }

}
