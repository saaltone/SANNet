package utils.matrix.operation;

import utils.matrix.Matrix;

import java.util.HashMap;

/**
 * Defines max pooling matrix operation.
 *
 */
public class MaxPoolMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input matrix.
     *
     */
    private Matrix input;

    /**
     * Number of inputs columns.
     *
     */
    private final int inputColumnSize;

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
     * Maximum position for each resulting row and column.
     *
     */
    private HashMap<Integer, Integer> maxPos;

    /**
     * Constructor for max pooling matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param inputColumnSize number of input columns.
     * @param filterRowSize filter size in rows.
     * @param filterColumnSize filter size in columns.
     */
    public MaxPoolMatrixOperation(int rows, int columns, int inputColumnSize, int filterRowSize, int filterColumnSize) {
        super(rows, columns, false);
        this.inputColumnSize = inputColumnSize;
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
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
        int maxRow = -1;
        int maxColumn = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int filterRow = 0; filterRow < filterRowSize; filterRow++) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn++) {
                int inputRow = row + filterRow;
                int inputColumn = column + filterColumn;
                double inputValue = input.getValue(inputRow, inputColumn);
                if (maxValue < inputValue) {
                    maxValue = inputValue;
                    maxRow = inputRow;
                    maxColumn = inputColumn;
                }
            }
        }
        result.setValue(row, column, maxValue);
        maxPos.put(2 * (row * inputColumnSize + column), maxRow);
        maxPos.put(2 * (row * inputColumnSize + column) + 1, maxColumn);
    }

}
