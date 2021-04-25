package utils.matrix.operation;

import utils.matrix.Matrix;

/**
 * Defines equal matrix operation.
 *
 */
public class EqualMatrixOperation extends AbstractMatrixOperation {

    /**
     * Other matrix.
     *
     */
    private Matrix other;

    /**
     * Constructor for equal matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     */
    public EqualMatrixOperation(int rows, int columns) {
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
     * Sets other matrix.
     *
     * @param other other matrix.
     */
    public void setOther(Matrix other) {
        this.other = other;
    }

    /**
     * Returns other matrix.
     *
     * @return other matrix.
     */
    public Matrix getOther() {
        return other;
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
