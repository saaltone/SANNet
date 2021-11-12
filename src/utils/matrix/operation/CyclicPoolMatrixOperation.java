package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;

/**
 * Defines cyclic pooling matrix operation.<br>
 * Traverses cyclically each filter row and column through step by step and propagates selected row and column.<br>
 *
 */
public class CyclicPoolMatrixOperation extends AbstractMatrixOperation {
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
     * Input position for each resulting row and column.
     *
     */
    private HashMap<Integer, Integer> inputPos;

    /**
     * Current row of filter.
     *
     */
    private transient int currentRow = 0;

    /**
     * Current column of filter.
     *
     */
    private transient int currentColumn = 0;

    /**
     * Constructor for random pooling matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param inputColumnSize number of input columns.
     * @param filterRowSize filter size in rows.
     * @param filterColumnSize filter size in columns.
     * @param stride stride step
     */
    public CyclicPoolMatrixOperation(int rows, int columns, int inputColumnSize, int filterRowSize, int filterColumnSize, int stride) {
        super(rows, columns, false, stride);
        this.inputColumnSize = inputColumnSize;
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
    }

    /**
     * Applies matrix operation.
     *
     * @param input input matrix.
     * @param inputPos input positions.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(Matrix input, HashMap<Integer, Integer> inputPos, Matrix result) throws MatrixException {
        this.input = input;
        this.inputPos = inputPos;
        this.result = result;
        applyMatrixOperation();
    }

    /**
     * Returns target matrix.
     *
     * @return target matrix.
     */
    protected Matrix getTargetMatrix() {
        return input;
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
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(int row, int column, double value) throws MatrixException {
        input.sliceAt(row, column, row + filterRowSize - 1, column + filterColumnSize - 1);
        int filterRow = currentRow;
        int filterColumn = currentColumn;
        double inputValue = input.getValue(filterRow, filterColumn);
        result.setValue(row, column, inputValue);
        inputPos.put(2 * (row * inputColumnSize + column), row + filterRow);
        inputPos.put(2 * (row * inputColumnSize + column) + 1, column + filterColumn);
        if(++currentRow >= filterRowSize) {
            currentRow = 0;
            if(++currentColumn >= filterColumnSize) currentColumn = 0;
        }
        input.unslice();
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
        input.sliceAt(row, column, row + filterRowSize - 1, column + filterColumnSize - 1);
        while (hasMaskAt(currentRow, currentColumn, input)) {
            if(++currentRow >= filterRowSize) {
                currentRow = 0;
                if(++currentColumn >= filterColumnSize) currentColumn = 0;
            }
        }
        int filterRow = currentRow;
        int filterColumn = currentColumn;
        double inputValue = input.getValue(filterRow, filterColumn);
        result.setValue(row, column, inputValue);
        inputPos.put(2 * (row * inputColumnSize + column), row + filterRow);
        inputPos.put(2 * (row * inputColumnSize + column) + 1, column + filterColumn);
        if(++currentRow >= filterRowSize) {
            currentRow = 0;
            if(++currentColumn >= filterColumnSize) currentColumn = 0;
        }
        input.unslice();
    }

}
