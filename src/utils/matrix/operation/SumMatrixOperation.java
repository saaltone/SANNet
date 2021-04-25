package utils.matrix.operation;

import utils.matrix.Matrix;

/**
 * Defines sum matrix operation.
 *
 */
public class SumMatrixOperation extends AbstractMatrixOperation {

    /**
     * Cumulated value.
     *
     */
    private double value;

    /**
     * Number of counted entries.
     *
     */
    private int count;

    /**
     * Constructor for sum matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     */
    public SumMatrixOperation(int rows, int columns) {
        super(rows, columns, true);
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
