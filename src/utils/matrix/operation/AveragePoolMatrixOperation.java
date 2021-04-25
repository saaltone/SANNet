package utils.matrix.operation;

import utils.matrix.Matrix;

/**
 * Defines average pooling matrix operation.
 *
 */
public class AveragePoolMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input matrix.
     *
     */
    private Matrix input;

    /**
     * Result.
     *
     */
    private Matrix result;

    /**
     * Number of rows in filter.
     *
     */
    private final int filterRowSize;

    /**
     * Number of columns in filter.
     *
     */
    private final int filterColumnSize;

    /**
     * Inverted size of filter = 1 / (rows * columns)
     *
     */
    private final double invertedFilterSize;

    /**
     * Constructor for average pooling matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param filterRowSize filter size in rows.
     * @param filterColumnSize filter size in columns.
     */
    public AveragePoolMatrixOperation(int rows, int columns, int filterRowSize, int filterColumnSize) {
        super(rows, columns, false);
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
        this.invertedFilterSize = 1 / (double)(filterRowSize * filterColumnSize);
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
     * Sets input matrix.
     *
     * @param input input matrix.
     */
    public void setInput(Matrix input) {
        this.input = input;
    }

    /**
     * Returns input matrix.
     *
     * @return input matrix.
     */
    public Matrix getInput() {
        return input;
    }

    /**
     * Sets result matrix.
     *
     * @param result result matrix.
     */
    public void setResult(Matrix result) {
        this.result = result;
    }

    /**
     * Returns result matrix.
     *
     * @return result matrix.
     */
    public Matrix getResult() {
        return result;
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
        for (int filterRow = 0; filterRow < filterRowSize; filterRow++) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn++) {
                sumValue += input.getValue(row + filterRow, column + filterColumn);
            }
        }
        result.setValue(row, column, sumValue * invertedFilterSize);
    }

}
