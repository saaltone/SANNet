/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.*;

/**
 * Implements matrix join operation.
 *
 */
public class JoinMatrixOperation extends AbstractMatrixOperation {

    /**
     * First matrix.
     *
     */
    private Matrix first;

    /**
     * Number of rows in first matrix.
     *
     */
    private int firstRows;

    /**
     * Number of columns in first matrix.
     *
     */
    private int firstColumns;

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
     * If true joins matrices vertically otherwise horizontally.
     *
     */
    private final boolean joinedVertically;

    /**
     * Constructor for join binary operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param joinedVertically if true joined vertically otherwise horizontally
     */
    public JoinMatrixOperation(int rows, int columns, boolean joinedVertically) {
        super(rows, columns, true);
        this.joinedVertically = joinedVertically;
    }

    /**
     * Applies matrix operation.
     *
     * @param first first matrix.
     * @param second second matrix.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(Matrix first, Matrix second, Matrix result) throws MatrixException {
        this.first = first;
        this.firstRows = first.getRows();
        this.firstColumns = first.getColumns();
        this.second = second;
        this.result = result;
        applyJoin(first, second, result);
    }

    /**
     * Joins two matrices either vertically or horizontally.
     *
     *
     * @param first first matrix
     * @param second second matrix
     * @param result result matrix
     * @throws MatrixException throws matrix exception if joining fails.
     */
    private void applyJoin(Matrix first, Matrix second, Matrix result) throws MatrixException {
        int firstRows = first.getRows();
        int firstColumns = first.getColumns();
        int secondRows = second.getRows();
        int secondColumns = second.getColumns();
        int resultRows = result.getRows();
        int resultColumns = result.getColumns();
        if (!(result instanceof JMatrix resultMatrix)) throw new MatrixException("Result must be of type JMatrix");
        if (resultMatrix.isJoinedVertically() != joinedVertically) throw new MatrixException("Result matrix must be joined " + (joinedVertically ? "vertically" : "horizontally"));
        if (joinedVertically ? firstColumns != secondColumns : firstRows != secondRows) throw new MatrixException("Merge " + (joinedVertically ? "Vertical" : "Horizontal") + " Incompatible matrix sizes: " + firstRows + "x" + firstColumns + " by " + secondRows + "x" + secondColumns);
        if (joinedVertically ? (resultRows != firstRows + secondRows || resultColumns != firstColumns) : (resultRows != firstRows || resultColumns != firstColumns + secondColumns)) throw new MatrixException("Dimensions of result matrix (" + resultRows + "x" + resultColumns + ") is not matching dimensions of " + (joinedVertically ? "vertically" : "horizontally") + " joined matrices (" + firstRows + "x" + firstColumns + "), (" + secondRows + "x" + secondColumns + ").");
        if (joinedVertically) {
            for (int column = 0; column < firstColumns; column++) {
                for (int row = 0; row < firstRows + secondRows; row++) {
                    result.setValue(row, column, row < firstRows ? first.getValue(row, column) : second.getValue(row - firstRows, column));
                }
            }
        }
        else {
            for (int row = 0; row < firstRows; row++) {
                for (int column = 0; column < firstColumns + secondColumns; column++) {
                    result.setValue(row, column, column < firstColumns ? first.getValue(row, column) : second.getValue(row, column - firstColumns));
                }
            }
        }
    }

    /**
     * Calculates gradient.
     *
     * @param first first matrix.
     * @param outputGradient output gradient.
     * @param firstMatrix if true gradient is returned for first matrix.
     * @return input gradient
     */
    public Matrix applyGradient(Matrix first, Matrix outputGradient, boolean firstMatrix) {
        int rows = joinedVertically ? firstMatrix ? first.getRows() : getRows() - first.getRows() : first.getRows();
        int columns = joinedVertically ? first.getColumns() : firstMatrix ? first.getColumns() : getColumns() - first.getColumns();
        Matrix result = new DMatrix(rows, columns);
        int firstRow = joinedVertically ? firstMatrix ? 0 : first.getRows() : 0;
        int firstColumn = joinedVertically ? 0 : firstMatrix ? 0 : first.getColumns();
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                result.setValue(row, column, outputGradient.getValue(firstRow + row, firstColumn + column));
            }
        }
        return result;
    }

    /**
     * Returns target matrix.
     *
     * @return target matrix.
     */
    protected Matrix getTargetMatrix() {
        return first;
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
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     */
    public void apply(int row, int column, double value) {
        if (joinedVertically) {
            result.setValue(row, column, row < firstRows ? first.getValue(row, column) : second.getValue(row - firstRows, column));
        }
        else {
            result.setValue(row, column, column < firstColumns ? first.getValue(row, column) : second.getValue(row, column - firstColumns));
        }
    }

}
