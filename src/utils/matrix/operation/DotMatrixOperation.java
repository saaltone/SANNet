package utils.matrix.operation;

import utils.matrix.Matrix;

/**
 * Defines dot operation.
 *
 */
public class DotMatrixOperation extends AbstractMatrixOperation {

    /**
     * First matrix.
     *
     */
    private Matrix first;

    /**
     * Second matrix.
     *
     */
    private Matrix second;

    /**
     * Result matrix.
     *
     */
    private Matrix result;

    /**
     * Constructor for dot matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     */
    public DotMatrixOperation(int rows, int columns) {
        super(rows, columns, false);
    }

    /**
     * Returns another matrix used in operation.
     *
     * @return another matrix used in operation.
     */
    public Matrix getAnother() {
        return second;
    }

    /**
     * Sets first matrix.
     *
     * @param first first matrix.
     */
    public void setFirst(Matrix first) {
        this.first = first;
    }

    /**
     * Returns first matrix.
     *
     * @return first matrix.
     */
    public Matrix getFirst() {
        return first;
    }

    /**
     * Sets second matrix.
     *
     * @param second second matrix.
     */
    public void setSecond(Matrix second) {
        this.second = second;
    }

    /**
     * Returns second matrix.
     *
     * @return second matrix.
     */
    public Matrix getSecond() {
        return second;
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
        for (int x = 0; x < first.getColumns(); x++) {
            result.setValue(row, column, result.getValue(row, column) + first.getValue(row, x) * second.getValue(x, column));
        }
    }

}
