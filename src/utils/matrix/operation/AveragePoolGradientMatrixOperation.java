package utils.matrix.operation;

import utils.matrix.Matrix;

/**
 * Defines average pooling gradient matrix operation.
 *
 */
public class AveragePoolGradientMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input gradient.
     *
     */
    private Matrix inputGradient;

    /**
     * Inverted size of pool 1 / (rows * columns)
     *
     */
    private final double invertedPoolSize;

    /**
     * Constructor for average pooling gradient matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param poolRows pool size in rows.
     * @param poolColumns pool size in columns.
     */
    public AveragePoolGradientMatrixOperation(int rows, int columns, int poolRows, int poolColumns) {
        super(rows, columns, false);
        this.invertedPoolSize = 1 / (double)(poolRows * poolColumns);
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
     * Sets input gradient matrix.
     *
     * @param inputGradient input gradient matrix.
     */
    public void setInputGradient(Matrix inputGradient) {
        this.inputGradient = inputGradient;
    }

    /**
     * Returns input gradient matrix.
     *
     * @return input gradient matrix.
     */
    public Matrix getInputGradient() {
        return inputGradient;
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     */
    public void apply(int row, int column, double value) {
        inputGradient.setValue(row, column, value * invertedPoolSize);
    }

}
