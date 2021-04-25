package utils.matrix.operation;

import utils.matrix.Matrix;

import java.util.HashMap;

/**
 * Defines max pooling gradient matrix operation.
 *
 */
public class MaxPoolGradientMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input gradient.
     *
     */
    private Matrix inputGradient;

    /**
     * Number of inputs columns.
     *
     */
    private final int inputColumnSize;

    /**
     * Maximum position for each resulting row and column.
     *
     */
    private HashMap<Integer, Integer> maxPos;

    /**
     * Constructor for max pooling gradient matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param inputColumnSize number of input columns.
     */
    public MaxPoolGradientMatrixOperation(int rows, int columns, int inputColumnSize) {
        super(rows, columns, true);
        this.inputColumnSize = inputColumnSize;
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
     * Sets maximum positions.
     *
     * @param maxPos maximum positions.
     */
    public void setMaxPos(HashMap<Integer, Integer> maxPos) {
        this.maxPos = maxPos;
    }

    /**
     * Returns maximum positions.
     *
     * @return maximum positions.
     */
    public HashMap<Integer, Integer> getMaxPos() {
        return maxPos;
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     */
    public void apply(int row, int column, double value) {
        inputGradient.setValue(maxPos.get(2 * (row * inputColumnSize + column)), maxPos.get(2 * (row * inputColumnSize + column) + 1), value);
    }

}
