package utils.matrix.operation;

import utils.matrix.Matrix;

/**
 * Defines norm matrix operation.
 *
 */
public class NormMatrixOperation extends AbstractMatrixOperation {

    /**
     * Power for norm operation.
     *
     */
    private final int p;

    /**
     * Cumulated norm value.
     *
     */
    private double value;

    /**
     * Constructor for norm matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param p power for norm operation.
     */
    public NormMatrixOperation(int rows, int columns, int p) {
        super(rows, columns, true);
        this.p = p;
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
