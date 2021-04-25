package utils.matrix.operation;

/**
 * Defines abstract matrix operation used by all matrix operations.
 *
 */
public abstract class AbstractMatrixOperation implements MatrixOperation {

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

}
