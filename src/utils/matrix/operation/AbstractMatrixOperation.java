package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;

/**
 * Defines abstract matrix operation used by all matrix operations.
 *
 */
public abstract class AbstractMatrixOperation implements MatrixOperation, Serializable {

    @Serial
    private static final long serialVersionUID = 4515327729821343316L;

    /**
     * Number of rows for operation.
     *
     */
    private final int rows;

    /**
     * Number of columns for operation.
     *
     */
    private final int columns;

    /**
     * If true operation provides value when applying operation otherwise false.
     *
     */
    private final boolean provideValue;

    /**
     * Constructor for abstract matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param provideValue if true operation provides value when applying operation otherwise false.
     */
    public AbstractMatrixOperation(int rows, int columns, boolean provideValue) {
        this.rows = rows;
        this.columns = columns;
        this.provideValue = provideValue;
    }

    /**
     * Returns number of rows for operation.
     *
     * @return number of rows for operation.
     */
    public int getRows() {
        return rows;
    }

    /**
     * Returns number of columns for operation.
     *
     * @return number of columns for operation.
     */
    public int getColumns() {
        return columns;
    }

    /**
     * If true operation provides value when applying operation otherwise false.
     *
     * @return if true operation provides value when applying operation otherwise false.
     */
    public boolean getProvideValue() {
        return provideValue;
    }

    /**
     * Applies operation assuming masked matrices.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyMask(int row, int column, double value) throws MatrixException {
        apply(row, column, value);
    }

    /**
     * Check if first matrix and optionally second matrix has mask.
     *
     * @param first first matrix.
     * @param second second matrix.
     * @return returns true if first or second matrix has mask.
     */
    public boolean hasMask(Matrix first, Matrix second) {
        return first.getMask() != null || (second != null && second.getMask() != null);
    }

    /**
     * Check if first matrix and optionally second matrix has mask at specific row and column.
     *
     * @param row row.
     * @param column column.
     * @param first first matrix.
     * @param second second matrix.
     * @return returns true if first or second matrix has mask at specific row and column.
     */
    public boolean hasMaskAt(int row, int column, Matrix first, Matrix second) {
        return first.hasMaskAt(row, column) || (second != null && second.hasMaskAt(row, column));
    }

    /**
     * Check if matrix has mask at specific row and column.
     *
     * @param row row.
     * @param column column.
     * @param matrix matrix.
     * @return returns true if first or second matrix has mask at specific row and column.
     */
    protected boolean hasMaskAt(int row, int column, Matrix matrix) {
        return matrix.hasMaskAt(row, column);
    }

}
