package utils.matrix.operation;

import utils.matrix.Matrix;

/**
 * Defines interface for matrix operation.
 *
 */
public interface MatrixOperation {

    /**
     * Returns number of rows for operation.
     *
     * @return number of rows for operation.
     */
    int getRows();

    /**
     * Returns number of columns for operation.
     *
     * @return number of columns for operation.
     */
    int getColumns();

    /**
     * Returns another matrix used in operation.
     *
     * @return another matrix used in operation.
     */
    Matrix getAnother();

    /**
     * If true operation provides value when applying operation otherwise false.
     *
     * @return true operation provides value when applying operation otherwise false.
     */
    boolean getProvideValue();

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     */
    void apply(int row, int column, double value);

}
