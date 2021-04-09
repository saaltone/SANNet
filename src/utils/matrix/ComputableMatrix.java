package utils.matrix;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

/**
 * Class that defines computable operations for matrices.
 *
 */
public abstract class ComputableMatrix extends AbstractMatrix {

    /**
     * Defines interface for matrix operation.
     *
     */
    private interface MatrixOperation {

        /**
         * Method to apply matrix operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        void apply(int row, int column, double value);
    }

    /**
     * Defines equal matrix operation.
     *
     */
    private static class EqualMatrixOperation implements MatrixOperation {

        /**
         * Other matrix.
         *
         */
        final Matrix other;

        /**
         * Constructor for equal operation.
         *
         * @param other other matrix.
         */
        EqualMatrixOperation(Matrix other) {
            this.other = other;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            other.setValue(row, column, value);
        }

    }

    /**
     * Defines matrix unary operation.
     *
     */
    private static class UnaryMatrixOperation implements MatrixOperation {

        /**
         * Result matrix.
         *
         */
        final Matrix result;

        /**
         * Matrix unary operation.
         *
         */
        final Matrix.MatrixUnaryOperation matrixUnaryOperation;

        /**
         * Constructor for matrix unary operation.
         *
         * @param result result matrix.
         * @param matrixUnaryOperation matrix unary operation.
         */
        UnaryMatrixOperation(Matrix result, Matrix.MatrixUnaryOperation matrixUnaryOperation) {
            this.result = result;
            this.matrixUnaryOperation = matrixUnaryOperation;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            result.setValue(row, column, matrixUnaryOperation.execute(value));
        }

    }

    /**
     * Defines matrix binary operation.
     *
     */
    private static class BinaryMatrixOperation implements MatrixOperation {

        /**
         * Other matrix.
         *
         */
        final Matrix other;

        /**
         * Result matrix.
         *
         */
        final Matrix result;

        /**
         * Matrix binary operation.
         *
         */
        final Matrix.MatrixBinaryOperation matrixBinaryOperation;

        /**
         * Constructor for matrix binary operation.
         *
         * @param other other matrix.
         * @param result result matrix.
         * @param matrixBinaryOperation matrix binary operation.
         */
        BinaryMatrixOperation(Matrix other, Matrix result, Matrix.MatrixBinaryOperation matrixBinaryOperation) {
            this.other = other;
            this.result = result;
            this.matrixBinaryOperation = matrixBinaryOperation;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            result.setValue(row, column, matrixBinaryOperation.execute(value, other.getValue(row, column)));
        }

    }

    private static class DotMatrixOperation implements MatrixOperation {

        /**
         * First matrix.
         *
         */
        final Matrix first;

        /**
         * Second matrix.
         *
         */
        final Matrix second;

        /**
         * Result matrix.
         *
         */
        final Matrix result;

        /**
         * Constructor for dot matrix operation.
         *
         * @param first first (this) matrix.
         * @param second second (other) matrix.
         * @param result result matrix.
         */
        public DotMatrixOperation(Matrix first, Matrix second, Matrix result) {
            this.first = first;
            this.second = second;
            this.result = result;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            for (int x = 0; x < first.getColumns(); x++) {
                result.setValue(row, column, result.getValue(row, column) + first.getValue(row, x) * second.getValue(x, column));
            }
        }

    }

    /**
     * Defines sum matrix operation.
     *
     */
    private static class SumMatrixOperation implements MatrixOperation {

        /**
         * Cumulated value.
         *
         */
        double value;

        /**
         * Number of counted entries.
         *
         */
        int count;

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            this.value += value;
            count++;
        }

        /**
         * Returns sum after operation is applied.
         *
         * @return sum.
         */
        public double getSum() {
            return value;
        }

        /**
         * Returns mean after operation is applied.
         *
         * @return mean.
         */
        public double getMean() {
            return value / (double)count;
        }

    }

    /**
     * Defines variance matrix operation.
     *
     */
    private static class VarianceMatrixOperation implements MatrixOperation {

        /**
         * Mean value for variance operation.
         *
         */
        final double mean;

        /**
         * Cumulated variance value.
         *
         */
        double value;

        /**
         * Number of counted entries.
         *
         */
        int count;

        /**
         * Constructor for variance operation.
         *
         * @param mean mean value for variance operation.
         */
        VarianceMatrixOperation(double mean) {
            this.mean = mean;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            this.value += Math.pow(value - mean, 2);
            count++;
        }

        /**
         * Returns variance after operation is applied.
         *
         * @return variance.
         */
        public double getVariance() {
            return count > 0 ? value / (double)count : 0;
        }

        /**
         * Returns standard deviation after operation is applied.
         *
         * @return standard deviation.
         */
        public double getStandardDeviation() {
            return count > 1 ? Math.sqrt(value / (double)(count - 1)) : 0;
        }

    }

    /**
     * Defines norm matrix operation.
     *
     */
    private static class NormMatrixOperation implements MatrixOperation {

        /**
         * Power for norm operation.
         *
         */
        final int p;

        /**
         * Cumulated norm value.
         *
         */
        double value;

        /**
         * Constructor for norm matrix operation.
         *
         * @param p power for norm operation.
         */
        NormMatrixOperation(int p) {
            this.p = p;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            this.value += Math.pow(Math.abs(value), p);
        }

        /**
         * Returns norm after operation is applied.
         *
         * @return norm.
         */
        public double getNorm() {
            return Math.pow(value, 1 / (double)p);
        }

    }

    /**
     * Defines normalize matrix operation.
     *
     */
    private static class NormalizeMatrixOperation implements MatrixOperation {

        /**
         * Mean for normalize operation.
         *
         */
        final double mean;

        /**
         * Variance for normalize operation.
         *
         */
        final double variance;

        /**
         * Cumulated norm value.
         *
         */
        final Matrix result;

        /**
         * Constructor for normalize matrix operation.
         *
         * @param mean mean for normalize operation.
         * @param variance variance for normalize operation.
         */
        NormalizeMatrixOperation(double mean, double variance, Matrix result) {
            this.mean = mean;
            this.variance = variance;
            this.result = result;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            result.setValue(row, column, (value - mean) / variance);
        }

    }

    /**
     * Defines minimum matrix operation.
     *
     */
    private static class MinMatrixOperation implements MatrixOperation {

        /**
         * Minimum value.
         *
         */
        double value = Double.POSITIVE_INFINITY;

        /**
         * Minimum row.
         *
         */
        int row = -1;

        /**
         * Minimum column.
         *
         */
        int column = -1;

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            if (value < this.value) {
                this.value = value;
                this.row = row;
                this.column = column;
            }
        }

    }

    /**
     * Defines maximum matrix operation.
     *
     */
    private static class MaxMatrixOperation implements MatrixOperation {

        /**
         * Maximum value.
         *
         */
        double value = Double.NEGATIVE_INFINITY;

        /**
         * Maximum row.
         *
         */
        int row = -1;

        /**
         * Maximum column.
         *
         */
        int column = -1;

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            if (value > this.value) {
                this.value = value;
                this.row = row;
                this.column = column;
            }
        }

    }

    /**
     * Defines Softmax gradient matrix operation.
     *
     */
    private static class SoftmaxGradientOperation implements MatrixOperation {

        /**
         * First matrix.
         *
         */
        final Matrix first;

        /**
         * Result matrix.
         *
         */
        final Matrix result;

        /**
         * Constructor for Softmax gradient matrix operation.
         *
         * @param result result matrix.
         */
        public SoftmaxGradientOperation(Matrix first, Matrix result) {
            this.first = first;
            this.result = result;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param row1 current row1.
         * @param value current value.
         */
        public void apply(int row, int row1, double value) {
            result.setValue(row1, row, (row == row1 ? 1 : 0) - first.getValue(row1, 0));
        }

    }

    /**
     * Defines abstract convolution matrix operation.
     *
     */
    private static abstract class AbstractConvolutionOperation implements MatrixOperation {

        /**
         * Input matrix.
         *
         */
        final Matrix input;

        /**
         * Filter matrix.
         *
         */
        final Matrix filter;

        /**
         * Result matrix.
         *
         */
        final Matrix result;

        /**
         * Matrix dilation value.
         *
         */
        final int dilation;

        /**
         * Filter row size.
         *
         */
        final int filterRowSize;

        /**
         * Filter column size.
         *
         */
        final int filterColumnSize;

        /**
         * Constructor for abstract convolution operation.
         *
         * @param input input
         * @param filter filter
         * @param result result
         * @param dilation dilation step
         */
        public AbstractConvolutionOperation(Matrix input, Matrix filter, Matrix result, int dilation) {
            this.input = input;
            this.filter = filter;
            this.result = result;
            this.dilation = dilation;
            this.filterRowSize = filter.getRows();
            this.filterColumnSize = filter.getColumns();
        }

    }

    /**
     * Defines crosscorrelation matrix operation.
     *
     */
    private static class CrosscorrelationOperation extends AbstractConvolutionOperation {

        /**
         * Constructor for crosscorrelation operation.
         *
         * @param input input
         * @param filter filter
         * @param result result
         * @param dilation dilation step
         */
        public CrosscorrelationOperation(Matrix input, Matrix filter, Matrix result, int dilation) {
            super(input, filter, result, dilation);
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            double resultValue = 0;
            for (int filterRow = 0; filterRow < filterRowSize; filterRow += dilation) {
                for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn += dilation) {
                    resultValue += input.getValue(row + filterRow, column + filterColumn) * filter.getValue(filterRow, filterColumn);
                }
            }
            result.setValue(row, column, resultValue);
        }

    }

    /**
     * Defines convolution matrix operation.
     *
     */
    private static class ConvolutionOperation extends AbstractConvolutionOperation {

        /**
         * Constructor for convolution operation.
         *
         * @param input input
         * @param filter filter
         * @param result result
         * @param dilation dilation step
         */
        public ConvolutionOperation(Matrix input, Matrix filter, Matrix result, int dilation) {
            super(input, filter, result, dilation);
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            double resultValue = 0;
            for (int filterRow = 0; filterRow < filterRowSize; filterRow += dilation) {
                for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn += dilation) {
                    resultValue += input.getValue(row + filterRow, column + filterColumn) * filter.getValue(filterRowSize - 1 - filterRow, filterColumnSize - 1 - filterColumn);
                }
            }
            result.setValue(row, column, resultValue);
        }

    }

    /**
     * Defines crosscorrelation input gradient operation.
     *
     */
    private static class CrosscorrelationInputGradientOperation implements MatrixOperation {

        /**
         * Filter matrix.
         *
         */
        final Matrix filter;

        /**
         * Number of rows in filter.
         *
         */
        final int filterRows;

        /**
         * Number of columns in filter.
         *
         */
        final int filterColumns;

        /**
         * Resulting input gradient.
         *
         */
        final Matrix inputGradient;

        /**
         * Dilation.
         *
         */
        final int dilation;

        /**
         * Constructor for crosscorrelation input gradient operation.
         *
         * @param filter filter
         * @param inputGradient input gradient
         * @param dilation dilation step
         */
        public CrosscorrelationInputGradientOperation(Matrix filter, Matrix inputGradient, int dilation) {
            this.filter = filter;
            filterRows = filter.getRows();
            filterColumns = filter.getColumns();
            this.inputGradient = inputGradient;
            this.dilation = dilation;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            for (int filterRow = 0; filterRow < filterRows; filterRow += dilation) {
                for (int filterColumn = 0; filterColumn < filterColumns; filterColumn += dilation) {
                    inputGradient.incrementByValue(row + filterRow, column + filterColumn, filter.getValue(filterRow, filterColumn) * value);
                }
            }
        }

    }

    /**
     * Defines convolution input gradient operation.
     *
     */
    private static class ConvolutionInputGradientOperation implements MatrixOperation {

        /**
         * Filter matrix.
         *
         */
        final Matrix filter;

        /**
         * Number of rows in filter.
         *
         */
        final int filterRows;

        /**
         * Number of columns in filter.
         *
         */
        final int filterColumns;

        /**
         * Resulting input gradient.
         *
         */
        final Matrix inputGradient;

        /**
         * Dilation.
         *
         */
        final int dilation;

        /**
         * Constructor for convolution input gradient operation.
         *
         * @param filter filter
         * @param inputGradient input gradient
         * @param dilation dilation step
         */
        public ConvolutionInputGradientOperation(Matrix filter, Matrix inputGradient, int dilation) {
            this.filter = filter;
            filterRows = filter.getRows();
            filterColumns = filter.getColumns();
            this.inputGradient = inputGradient;
            this.dilation = dilation;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            for (int filterRow = 0; filterRow < filterRows; filterRow += dilation) {
                for (int filterColumn = 0; filterColumn < filterColumns; filterColumn += dilation) {
                    inputGradient.incrementByValue(row + filterRow, column + filterColumn, filter.getValue(filterRows - 1 - filterRow, filterColumns - 1 - filterColumn) * value);
                }
            }
        }

    }

    /**
     * Defines crosscorrelation filter gradient operation.
     *
     */
    private static class CrosscorrelationFilterGradientOperation implements MatrixOperation {

        /**
         * Input matrix.
         *
         */
        final Matrix input;

        /**
         * Number of rows in filter.
         *
         */
        final int filterRows;

        /**
         * Number of columns in filter.
         *
         */
        final int filterColumns;

        /**
         * Resulting filter gradient.
         *
         */
        final Matrix filterGradient;

        /**
         * Dilation.
         *
         */
        final int dilation;

        /**
         * Constructor for crosscorrelation filter gradient operation.
         *
         * @param input input
         * @param filterGradient input gradient
         * @param dilation dilation step
         */
        public CrosscorrelationFilterGradientOperation(Matrix input, Matrix filterGradient, int dilation) {
            this.input = input;
            filterRows = filterGradient.getRows();
            filterColumns = filterGradient.getColumns();
            this.filterGradient = filterGradient;
            this.dilation = dilation;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            for (int filterRow = 0; filterRow < filterRows; filterRow += dilation) {
                for (int filterColumn = 0; filterColumn < filterColumns; filterColumn += dilation) {
                    filterGradient.incrementByValue(filterRow, filterColumn, input.getValue(row + filterRow, column + filterColumn) * value);
                }
            }
        }

    }

    /**
     * Defines convolution filter gradient operation.
     *
     */
    private static class ConvolutionFilterGradientOperation implements MatrixOperation {

        /**
         * Input matrix.
         *
         */
        final Matrix input;

        /**
         * Number of rows in filter.
         *
         */
        final int filterRows;

        /**
         * Number of columns in filter.
         *
         */
        final int filterColumns;

        /**
         * Resulting filter gradient.
         *
         */
        final Matrix filterGradient;

        /**
         * Dilation.
         *
         */
        final int dilation;

        /**
         * Constructor for convolution filter gradient operation.
         *
         * @param input input
         * @param filterGradient input gradient
         * @param dilation dilation step
         */
        public ConvolutionFilterGradientOperation(Matrix input, Matrix filterGradient, int dilation) {
            this.input = input;
            filterRows = filterGradient.getRows();
            filterColumns = filterGradient.getColumns();
            this.filterGradient = filterGradient;
            this.dilation = dilation;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            for (int filterRow = 0; filterRow < filterRows; filterRow += dilation) {
                for (int filterColumn = 0; filterColumn < filterColumns; filterColumn += dilation) {
                    filterGradient.incrementByValue(filterRows - 1 - filterRow, filterColumns - 1 - filterColumn, input.getValue(row + filterRow, column + filterColumn) * value);
                }
            }
        }

    }

    /**
     * Defines max pooling operation.
     *
     */
    private static class MaxPoolOperation implements MatrixOperation {

        /**
         * Input matrix.
         *
         */
        final Matrix input;

        /**
         * Number of inputs rows.
         *
         */
        final int inputRows;

        /**
         * Number of inputs columns.
         *
         */
        final int inputColumns;

        /**
         * Result.
         *
         */
        final Matrix result;

        /**
         * Number of rows in pool.
         *
         */
        final int poolRows;

        /**
         * Number of columns in pool.
         *
         */
        final int poolColumns;

        /**
         * Maximum position for each resulting row and column.
         *
         */
        final HashMap<Integer, Integer> maxPos;

        /**
         * Constructor for max pooling operation.
         *
         * @param input input
         * @param poolRows pool size in rows.
         * @param poolColumns pool size in columns.
         * @param maxPos maximum position for each resulting row and column..
         */
        public MaxPoolOperation(Matrix input, Matrix result, int poolRows, int poolColumns, HashMap<Integer, Integer> maxPos) {
            this.input = input;
            this.inputRows = input.getRows();
            this.inputColumns = input.getColumns();
            this.result = result;
            this.poolRows = poolRows;
            this.poolColumns = poolColumns;
            this.maxPos = maxPos;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            int maxRow = -1;
            int maxColumn = -1;
            double maxValue = Double.NEGATIVE_INFINITY;
            for (int poolRow = 0; poolRow < poolRows; poolRow++) {
                for (int poolColumn = 0; poolColumn < poolColumns; poolColumn++) {
                    int inputRow = row + poolRow;
                    int inputColumn = column + poolColumn;
                    double inputValue = input.getValue(inputRow, inputColumn);
                    if (maxValue < inputValue) {
                        maxValue = inputValue;
                        maxRow = inputRow;
                        maxColumn = inputColumn;
                    }
                }
            }
            result.setValue(row, column, maxValue);
            maxPos.put(2 * (row * inputColumns + column), maxRow);
            maxPos.put(2 * (row * inputColumns + column) + 1, maxColumn);
        }

    }

    /**
     * Defines max pooling gradient operation.
     *
     */
    private static class MaxPoolGradientOperation implements MatrixOperation {

        /**
         * Input gradient.
         *
         */
        final Matrix inputGradient;

        /**
         * Number of inputs rows.
         *
         */
        final int inputRows;

        /**
         * Number of inputs columns.
         *
         */
        final int inputColumns;

        /**
         * Maximum position for each resulting row and column.
         *
         */
        final HashMap<Integer, Integer> maxPos;

        /**
         * Constructor for max pooling gradient operation.
         *
         * @param inputGradient input gradient
         * @param maxPos maximum positions for row and column.
         */
        public MaxPoolGradientOperation(Matrix inputGradient, HashMap<Integer, Integer> maxPos) {
            this.inputGradient = inputGradient;
            this.inputRows = inputGradient.getRows();
            this.inputColumns = inputGradient.getColumns();
            this.maxPos = maxPos;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            inputGradient.setValue(maxPos.get(2 * (row * inputColumns + column)), maxPos.get(2 * (row * inputColumns + column) + 1), value);
        }

    }

    /**
     * Defines average pooling operation.
     *
     */
    private static class AveragePoolOperation implements MatrixOperation {

        /**
         * Input matrix.
         *
         */
        final Matrix input;

        /**
         * Resulting filter gradient.
         *
         */
        final Matrix result;

        /**
         * Number of rows in pool.
         *
         */
        final int poolRows;

        /**
         * Number of columns in pool.
         *
         */
        final int poolColumns;

        /**
         * Inverted size of pool 1 / (rows * columns)
         *
         */
        final double invertedPoolSize;

        /**
         * Constructor for average pooling operation.
         *
         * @param input input
         * @param poolRows pool size in rows.
         * @param poolColumns pool size in columns.
         */
        public AveragePoolOperation(Matrix input, Matrix result, int poolRows, int poolColumns) {
            this.input = input;
            this.result = result;
            this.poolRows = poolRows;
            this.poolColumns = poolColumns;
            this.invertedPoolSize = 1 / (double)(poolRows * poolColumns);
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            double sumValue = 0;
            for (int poolRow = 0; poolRow < poolRows; poolRow++) {
                for (int poolColumn = 0; poolColumn < poolColumns; poolColumn++) {
                    sumValue += input.getValue(row + poolRow, column + poolColumn);
                }
            }
            result.setValue(row, column, sumValue * invertedPoolSize);
        }

    }

    /**
     * Defines average pooling gradient operation.
     *
     */
    private static class AveragePoolGradientOperation implements MatrixOperation {

        /**
         * Input gradient.
         *
         */
        final Matrix inputGradient;

        /**
         * Inverted size of pool 1 / (rows * columns)
         *
         */
        final double invertedPoolSize;

        /**
         * Constructor for average pooling gradient operation.
         *
         * @param inputGradient input gradient
         * @param poolRows pool size in rows.
         * @param poolColumns pool size in columns.
         */
        public AveragePoolGradientOperation(Matrix inputGradient, int poolRows, int poolColumns) {
            this.inputGradient = inputGradient;
            this.invertedPoolSize = 1 / (double)(poolRows * poolColumns);
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            inputGradient.setValue(row, column, value * invertedPoolSize);
        }

    }

    /**
     * Stride size for convolutional and pooling operations.
     *
     */
    private int stride = 1;

    /**
     * Dilation step size for convolutional operations.
     *
     */
    private int dilation = 1;

    /**
     * Filter size for convolutional operations.
     *
     */
    private int filterSize = 3;

    /**
     * Pool size for pooling operations.
     *
     */
    private int poolSize = 2;

    /**
     * Random function for matrix class.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for matrix.
     *
     * @param isScalar true if matrix is scalar (size 1x1).
     */
    protected ComputableMatrix(boolean isScalar) {
        super(isScalar);
    }

    /**
     * Constructor for matrix.
     *
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param name name if matrix.
     */
    protected ComputableMatrix(boolean isScalar, String name) {
        super(isScalar, name);
    }

    /**
     * Checks if data of other matrix is equal to data of this matrix
     *
     * @param other matrix to be compared.
     * @return true is data of this and other matrix are equal otherwise false.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public boolean equals(Matrix other) throws MatrixException {
        if (other.getRows() != getRows() || other.getColumns() != getColumns()) {
            throw new MatrixException("Incompatible target matrix size: " + other.getRows() + "x" + other.getColumns());
        }

        for (int row = 0; row < other.getRows(); row++) {
            for (int column = 0; column < other.getColumns(); column++) {
                if (getValue(row, column) != other.getValue(row, column)) return false;
            }
        }
        return true;
    }

    /**
     * Makes current matrix data equal to other matrix data.
     *
     * @param other other matrix to be copied as data of this matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void setEqualTo(Matrix other) throws MatrixException {
        if (other.getRows() != getRows() || other.getColumns() != getColumns()) {
            throw new MatrixException("Incompatible target matrix size: " + other.getRows() + "x" + other.getColumns());
        }

        EqualMatrixOperation equalMatrixOperation = new EqualMatrixOperation(other);
        // Ignores masking of other matrix.
        applyMatrixOperation(equalMatrixOperation, null, getRows(), getColumns(), true);

    }

    /**
     * Applies single variable operation to this matrix and stores operation result into result matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param result matrix which stores operation result.
     * @param matrixUnaryOperation single variable operation defined as lambda operator.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and result matrix are not of equal dimensions.
     */
    public Matrix apply(Matrix result, Matrix.MatrixUnaryOperation matrixUnaryOperation) throws MatrixException {
        if (result.getRows() != getRows() || result.getColumns() != getColumns()) {
            throw new MatrixException("Incompatible result matrix sizes: " + result.getRows() + "x" + result.getColumns());
        }

        UnaryMatrixOperation unaryMatrixOperation = new UnaryMatrixOperation(result, matrixUnaryOperation);
        applyMatrixOperation(unaryMatrixOperation, null, getRows(), getColumns(), true);

        return result;
    }

    /**
     * Applies two variable operation to this matrix and other matrix and stores operation result into result matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix or
     * multiplying current matrix with other matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @param matrixBinaryOperation two variable operation defined as lambda operator.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     */
    public Matrix applyBi(Matrix other, Matrix result, Matrix.MatrixBinaryOperation matrixBinaryOperation) throws MatrixException {
        if (!isScalar() && !other.isScalar() && (getRows() != other.getRows() || getColumns() != other.getColumns())) {
            throw new MatrixException("Incompatible matrix sizes: " + getRows() + "x" + getColumns() + " by " + other.getRows() + "x" + other.getColumns());
        }
        if (!isScalar() && !result.isScalar() && (getRows() != result.getRows() || getColumns() != result.getColumns())) {
            throw new MatrixException("Incompatible result matrix sizes: " + result.getRows() + "x" + result.getColumns());
        }

        // Checks if there is need to broadcast or un-broadcast due to scalar matrix.
        int rows = !isScalar() ? getRows() : other.getRows();
        int columns = !isScalar() ? getColumns() : other.getColumns();

        BinaryMatrixOperation binaryMatrixOperation = new BinaryMatrixOperation(other, result, matrixBinaryOperation);
        applyMatrixOperation(binaryMatrixOperation, other, rows, columns, true);

        return result;
    }

    /**
     * Takes matrix dot product of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if columns of this matrix and rows of other matrix are not matching or rows of this and result matrix or columns of result and other matrix are not matching.
     */
    protected void applyDot(Matrix other, Matrix result) throws MatrixException {
        if (getColumns() != other.getRows()) {
            throw new MatrixException("Incompatible matrix sizes: " + getRows() + "x" + getColumns() + " by " + other.getRows() + "x" + other.getColumns());
        }
        if (getRows() != result.getRows() || other.getColumns() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }

        DotMatrixOperation dotMatrixOperation = new DotMatrixOperation(this, other, result);
        applyMatrixOperation(dotMatrixOperation, other, getRows(), other.getColumns(), false);

    }

    /**
     * Applies matrix operation.
     *
     * @param matrixOperation matrix operation.
     */
    private void applyMatrixOperation(MatrixOperation matrixOperation) {
        applyMatrixOperation(matrixOperation, null, getRows(), getColumns(), true);
    }

    /**
     * Applies matrix operation.
     *
     * @param matrixOperation matrix operation.
     * @param rows number of matrix rows.
     * @param columns number of matrix columns.
     * @param provideValue if true value will be provided for matrix operation otherwise zero (no value) if provided.
     */
    private void applyMatrixOperation(MatrixOperation matrixOperation, Matrix other, int rows, int columns, final boolean provideValue) {
        final int rowStride = stride;
        final int columnStride = stride;
        if (!hasMask(other)) {
            for (int row = 0; row < rows; row += rowStride) {
                for (int column = 0; column < columns; column += columnStride) {
                    matrixOperation.apply(row, column, provideValue ? getValue(row, column) : 0);
                }
            }
        }
        else {
            for (int row = 0; row < rows; row += rowStride) {
                if (!hasRowMaskAt(row, other)) {
                    for (int column = 0; column < columns; column += columnStride) {
                        if (!hasMaskAt(row, column, other) && !hasColumnMaskAt(column, other)) {
                            matrixOperation.apply(row, column, provideValue ? getValue(row, column) : 0);
                        }
                    }
                }
            }
        }
    }

    /**
     * Check if matrix and optionally other matrix has mask.
     *
     * @param other other matrix.
     * @return returns true if this or other matrix has mask.
     */
    private boolean hasMask(Matrix other) {
        return getMask() != null || (other != null && other.getMask() != null);
    }

    /**
     * Check if matrix and optionally other matrix has mask at specific row and column.
     *
     * @param row row.
     * @param column column.
     * @param other other matrix.
     * @return returns true if this or other matrix has mask at specific row and column.
     */
    private boolean hasMaskAt(int row, int column, Matrix other) {
        return hasMaskAt(this, row, column) || (other != null && hasMaskAt(other, row, column));
    }

    /**
     * Check if matrix and optionally other matrix has mask at specific row.
     *
     * @param row row.
     * @param other other matrix.
     * @return returns true if this or other matrix has mask at specific row.
     */
    private boolean hasRowMaskAt(int row, Matrix other) {
        return hasRowMaskAt(this, row) || (other != null && hasRowMaskAt(other, row));
    }

    /**
     * Check if matrix and optionally other matrix has mask at specific column.
     *
     * @param column column.
     * @param other other matrix.
     * @return returns true if this or other matrix has mask at specific column.
     */
    private boolean hasColumnMaskAt(int column, Matrix other) {
        return hasColumnMaskAt(this, column) || (other != null && hasColumnMaskAt(other, column));
    }

    /**
     * Takes element wise cumulative sum of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return cumulative sum of this matrix.
     */
    public double sum() {
        SumMatrixOperation sumMatrixOperation = new SumMatrixOperation();
        applyMatrixOperation(sumMatrixOperation);
        return sumMatrixOperation.getSum();
    }

    /**
     * Takes mean of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return mean of elements of this matrix.
     */
    public double mean() {
        SumMatrixOperation sumMatrixOperation = new SumMatrixOperation();
        applyMatrixOperation(sumMatrixOperation);
        return sumMatrixOperation.getMean();
    }

    /**
     * Takes variance of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return variance of elements of this matrix.
     */
    public double variance(double mean) {
        VarianceMatrixOperation varianceMatrixOperation = new VarianceMatrixOperation(mean);
        applyMatrixOperation(varianceMatrixOperation);
        return varianceMatrixOperation.getVariance();
    }

    /**
     * Takes standard deviation of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return standard deviation of elements of this matrix.
     */
    public double standardDeviation(double mean) {
        VarianceMatrixOperation varianceMatrixOperation = new VarianceMatrixOperation(mean);
        applyMatrixOperation(varianceMatrixOperation);
        return varianceMatrixOperation.getStandardDeviation();
    }

    /**
     * Takes cumulative p- norm (p is number equal or bigger than 1) of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param p p value for norm.
     * @return cumulative norm value of matrix.
     */
    public double norm(int p) {
        NormMatrixOperation normMatrixOperation = new NormMatrixOperation(p);
        applyMatrixOperation(normMatrixOperation);
        return normMatrixOperation.getNorm();
    }

    /**
     * Normalizes matrix by removing mean and variance.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @param inplace if true matrix is normalized in place otherwise copy of normalized matrix is returned.
     * @return normalized matrix.
     */
    public Matrix normalize(boolean inplace) {
        Matrix result = inplace ? this : getNewMatrix();
        NormalizeMatrixOperation normalizeMatrixOperation = new NormalizeMatrixOperation(mean(), variance(), result);
        applyMatrixOperation(normalizeMatrixOperation);
        return result;
    }

    /**
     * Returns minimum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return minimum value of matrix.
     */
    public double min() {
        MinMatrixOperation minMatrixOperation = new MinMatrixOperation();
        applyMatrixOperation(minMatrixOperation);
        return minMatrixOperation.value;
    }

    /**
     * Returns argmin meaning row and column of matrix containing minimum value.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return array containing row and column in this order that points to minimum value of matrix.
     */
    public int[] argmin() {
        MinMatrixOperation minMatrixOperation = new MinMatrixOperation();
        applyMatrixOperation(minMatrixOperation);
        int[] result = new int[2];
        result[0] = minMatrixOperation.row;
        result[1] = minMatrixOperation.column;
        return result;
    }

    /**
     * Returns maximum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return maximum value of matrix.
     */
    public double max() {
        MaxMatrixOperation maxMatrixOperation = new MaxMatrixOperation();
        applyMatrixOperation(maxMatrixOperation);
        return maxMatrixOperation.value;
    }

    /**
     * Returns argmax meaning row and column of matrix containing maximum value.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return array containing row and column in this order that points to maximum value of matrix.
     */
    public int[] argmax() {
        MaxMatrixOperation maxMatrixOperation = new MaxMatrixOperation();
        applyMatrixOperation(maxMatrixOperation);
        int[] result = new int[2];
        result[0] = maxMatrixOperation.row;
        result[1] = maxMatrixOperation.column;
        return result;
    }

    /**
     * Returns softmax of this matrix.
     *
     * @param result result matrix.
     * @return softmax of this matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmax(Matrix result) throws MatrixException {
        if (getColumns() != 1) {
            throw new MatrixException("Matrix must be a column vector.");
        }
        if (getRows() != result.getRows() || getColumns() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }

        final double maxValue = max();
        apply(result, (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.exp(value - maxValue));
        result.divide(result.sum(), result);

        return result;
    }

    /**
     * Returns Gumbel softmax of this matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @param result result matrix.
     * @param gumbelSoftmaxTau tau value for Gumbel Softmax.
     * @return softmax of this matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix gumbelSoftmax(Matrix result, double gumbelSoftmaxTau) throws MatrixException {
        if (getColumns() != 1) {
            throw new MatrixException("Matrix must be a column vector.");
        }
        if (getRows() != result.getRows() || getColumns() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }

        apply(result, (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.exp((Math.log(Math.exp(value) / (1 + Math.exp(value))) + getGumbelNoise()) / gumbelSoftmaxTau));
        result.divide(result.sum(), result);

        return result;
    }

    /**
     * Returns Gumbel noise.<br>
     *
     * @return Gumbel noise.
     */
    private double getGumbelNoise() {
        double epsilon = 10E-8;
        return -Math.log(-Math.log(random.nextDouble() + epsilon) + epsilon);
    }

    /**
     * Returns softmax gradient of this matrix.<br>
     * Assumes that input matrix is softmax result.<br>
     *
     * @param result result matrix.
     * @return softmax gradient of this matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmaxGrad(Matrix result) throws MatrixException {
        if (getColumns() != 1) {
            throw new MatrixException("Matrix must be a column vector.");
        }
        if (getRows() != result.getRows() || getRows() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }

        SoftmaxGradientOperation softmaxGradientOperation = new SoftmaxGradientOperation(this, result);
        applyMatrixOperation(softmaxGradientOperation, null, getRows(), getRows(), false);

        return result;
    }

    /**
     * Sets stride size for convolution and pooling operations.
     *
     * @param stride stride size.
     */
    public void setStride(int stride) {
        this.stride = stride;
    }

    /**
     * Returns stride size for convolution and pooling operations.
     *
     * @return stride size.
     */
    public int getStride() {
        return stride;
    }

    /**
     * Sets dilation step size for convolution operations.
     *
     * @param dilation dilation step size.
     */
    public void setDilation(int dilation) {
        this.dilation = dilation;
    }

    /**
     * Returns dilation step size for convolution operations.
     *
     * @return dilation step size.
     */
    public int getDilation() {
        return dilation;
    }

    /**
     * Sets filter size for convolution operations.
     *
     * @param filterSize filter size.
     */
    public void setFilterSize(int filterSize) {
        this.filterSize = filterSize;
    }

    /**
     * Returns filter size.
     *
     * @return filter size
     */
    public int getFilterSize() {
        return filterSize;
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated result of convolution.
     */
    protected void applyConvolve(Matrix filter, Matrix result) {
        ConvolutionOperation convolutionOperation = new ConvolutionOperation(this, filter, result, dilation);
        applyMatrixOperation(convolutionOperation, null, result.getRows(), result.getColumns(), false);
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated result of convolution.
     */
    protected void applyCrosscorrelate(Matrix filter, Matrix result) {
        CrosscorrelationOperation crosscorrelationOperation = new CrosscorrelationOperation(this, filter, result, dilation);
        applyMatrixOperation(crosscorrelationOperation, null, result.getRows(), result.getColumns(), false);
    }

    /**
     * Calculates gradient of convolution for input.
     *
     * @param filter filter for convolution operator.
     * @param inputGradient input gradient.
     */
    public void convolveInputGradient(Matrix filter, Matrix inputGradient) {
        ConvolutionInputGradientOperation convolutionInputGradientOperation = new ConvolutionInputGradientOperation(filter, inputGradient, dilation);
        applyMatrixOperation(convolutionInputGradientOperation, null, getRows(), getColumns(), true);
    }

    /**
     * Calculates gradient of crosscorrelation for input.
     *
     * @param filter filter for crosscorrelation operator.
     * @param inputGradient input gradient.
     */
    public void crosscorrelateInputGradient(Matrix filter, Matrix inputGradient) {
        CrosscorrelationInputGradientOperation crosscorrelationInputGradientOperation = new CrosscorrelationInputGradientOperation(filter, inputGradient, dilation);
        applyMatrixOperation(crosscorrelationInputGradientOperation, null, getRows(), getColumns(), true);
    }

    /**
     * Calculates gradient of convolution for filter.
     *
     * @param input input for convolutional operator.
     * @param filterGradient result gradient.
     */
    public void convolveFilterGradient(Matrix input, Matrix filterGradient) {
        ConvolutionFilterGradientOperation convolutionFilterGradientOperation = new ConvolutionFilterGradientOperation(input, filterGradient, dilation);
        applyMatrixOperation(convolutionFilterGradientOperation, null, getRows(), getColumns(), true);
    }

    /**
     * Calculates gradient of crosscorrelation for filter.
     *
     * @param input input for crosscorrelation operator.
     * @param filterGradient result gradient.
     */
    public void crosscorrelateFilterGradient(Matrix input, Matrix filterGradient) {
        CrosscorrelationFilterGradientOperation crosscorrelationFilterGradientOperation = new CrosscorrelationFilterGradientOperation(input, filterGradient, dilation);
        applyMatrixOperation(crosscorrelationFilterGradientOperation, null, getRows(), getColumns(), true);
    }

    /**
     * Sets size of pool for pooling operation.
     *
     * @param poolSize pool size.
     */
    public void setPoolSize(int poolSize) {
        this.poolSize = poolSize;
    }

    /**
     * Returns pool size.
     *
     * @return pool size.
     */
    public int getPoolSize() {
        return poolSize;
    }

    /**
     * Calculates max pooling operation for this matrix and returns max arguments.
     *
     * @param result result matrix.
     * @param maxPos maximum position for each result row and column value.
     */
    protected void applyMaxPool(Matrix result, HashMap<Integer, Integer> maxPos) {
        MaxPoolOperation maxPoolOperation = new MaxPoolOperation(this, result, poolSize, poolSize, maxPos);
        applyMatrixOperation(maxPoolOperation, null, result.getRows(), result.getColumns(), false);
    }

    /**
     * Calculates gradient for max pool operation.
     *
     * @param inputGradient input gradient.
     * @param maxPos maximum position for each result row and column value.
     */
    public void maxPoolGradient(Matrix inputGradient, HashMap<Integer, Integer> maxPos) {
        MaxPoolGradientOperation maxPoolGradientOperation = new MaxPoolGradientOperation(inputGradient, maxPos);
        applyMatrixOperation(maxPoolGradientOperation, null, getRows(), getColumns(), true);
    }

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @param result result matrix.
     */
    protected void applyAveragePool(Matrix result) {
        AveragePoolOperation averagePoolOperation = new AveragePoolOperation(this, result, poolSize, poolSize);
        applyMatrixOperation(averagePoolOperation, null, result.getRows(), result.getColumns(), false);
    }

    /**
     * Calculates gradient of average pooling operation for this matrix.
     *
     * @param inputGradient input gradient.
     */
    public void averagePoolGradient(Matrix inputGradient) {
        AveragePoolGradientOperation averagePoolGradientOperation = new AveragePoolGradientOperation(inputGradient, poolSize, poolSize);
        applyMatrixOperation(averagePoolGradientOperation, null, getRows(), getColumns(), false);
    }

}
