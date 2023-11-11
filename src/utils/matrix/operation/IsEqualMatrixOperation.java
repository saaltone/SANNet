/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements matrix operation to check if two matrices are equal.
 *
 */
public class IsEqualMatrixOperation extends AbstractMatrixOperation {

    /**
     * Constructor for is equal matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     */
    public IsEqualMatrixOperation(int rows, int columns, int depth) {
        super(rows, columns, depth, true);
    }

    /**
     * Applies matrix operation.
     *
     * @param first first matrix.
     * @param second second matrix.
     * @return true is data of this and other matrix are equal otherwise false.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public boolean apply(Matrix first, Matrix second) throws MatrixException {
        int rows = getRows();
        int columns = getColumns();
        int totalDepth = getDepth();
        int otherRows = second.getRows();
        int otherColumns = second.getColumns();
        int otherTotalDepth = second.getDepth();
        if (otherRows != rows || otherColumns != columns || otherTotalDepth != totalDepth) {
            throw new MatrixException("Incompatible target matrix size: " + otherRows + "x" + otherColumns + "x" + otherTotalDepth);
        }

        for (int depth = 0; depth < totalDepth; depth++) {
            for (int row = 0; row < otherRows; row++) {
                for (int column = 0; column < otherColumns; column++) {
                    if (first.getValue(row, column, depth) != second.getValue(row, column, depth)) return false;
                }
            }
        }
        return true;
    }

    /**
     * Applies operation.<br>
     * Ignores masking of other matrix.<br>
     *
     * @param row    current row.
     * @param column current column.
     * @param depth  current depth.
     * @param value  current value.
     * @param result result matrix.
     */
    public void apply(int row, int column, int depth, double value, Matrix result) {
    }

}
