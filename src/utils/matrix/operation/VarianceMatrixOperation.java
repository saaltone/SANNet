package utils.matrix.operation;

import utils.matrix.Matrix;

/**
 * Defines variance matrix operation.
 *
 */
public class VarianceMatrixOperation extends AbstractMatrixOperation {

    /**
     * Mean value for variance operation.
     *
     */
    private final double mean;

    /**
     * Cumulated variance value.
     *
     */
    private double value;

    /**
     * Number of counted entries.
     *
     */
    private int count;

    /**
     * Constructor for variance operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param mean mean value for variance operation.
     */
    public VarianceMatrixOperation(int rows, int columns, double mean) {
        super(rows, columns, true);
        this.mean = mean;
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
