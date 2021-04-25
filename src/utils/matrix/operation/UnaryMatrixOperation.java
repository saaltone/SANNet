package utils.matrix.operation;

import utils.matrix.Matrix;

/**
 * Defines matrix unary operation.
 *
 */
public class UnaryMatrixOperation extends AbstractMatrixOperation {

    /**
     * Result matrix.
     *
     */
    private Matrix result;

    /**
     * Matrix unary operation.
     *
     */
    private final Matrix.MatrixUnaryOperation matrixUnaryOperation;

    /**
     * Constructor for matrix unary operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param matrixUnaryOperation matrix unary operation.
     */
    public UnaryMatrixOperation(int rows, int columns, Matrix.MatrixUnaryOperation matrixUnaryOperation) {
        super(rows, columns, true);
        this.matrixUnaryOperation = matrixUnaryOperation;
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
        result.setValue(row, column, matrixUnaryOperation.execute(value));
    }

}
