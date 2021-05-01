package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

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
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void apply(int row, int column, double value) throws MatrixException;

    /**
     * Applies operation assuming masked matrices.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void applyMask(int row, int column, double value) throws MatrixException;

    /**
     * Check if first matrix and optionally second matrix has mask.
     *
     * @param first first matrix.
     * @param second second matrix.
     * @return returns true if first or second matrix has mask.
     */
    boolean hasMask(Matrix first, Matrix second);

    /**
     * Check if first matrix and optionally second matrix has mask at specific row and column.
     *
     * @param row row.
     * @param column column.
     * @param first first matrix.
     * @param second second matrix.
     * @return returns true if first or second matrix has mask at specific row and column.
     */
    boolean hasMaskAt(int row, int column, Matrix first, Matrix second);

}
