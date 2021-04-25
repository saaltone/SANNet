package utils.matrix.operation;

import utils.matrix.Matrix;

/**
 * Defines matrix binary operation.
 *
 */
public class BinaryMatrixOperation extends AbstractMatrixOperation {

    /**
     * Other matrix.
     *
     */
    private Matrix other;

    /**
     * Result matrix.
     *
     */
    private Matrix result;

    /**
     * Matrix binary operation.
     *
     */
    private final Matrix.MatrixBinaryOperation matrixBinaryOperation;

    /**
     * Constructor for matrix binary operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param matrixBinaryOperation matrix binary operation.
     */
    public BinaryMatrixOperation(int rows, int columns, Matrix.MatrixBinaryOperation matrixBinaryOperation) {
        super(rows, columns, true);
        this.matrixBinaryOperation = matrixBinaryOperation;
    }

    /**
     * Returns another matrix used in operation.
     *
     * @return another matrix used in operation.
     */
    public Matrix getAnother() {
        return other;
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
        result.setValue(row, column, matrixBinaryOperation.execute(value, other.getValue(row, column)));
    }

}
