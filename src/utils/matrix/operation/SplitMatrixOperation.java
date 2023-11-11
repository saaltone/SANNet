/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.*;

import java.util.ArrayList;

/**
 * Implements split matrix operation.
 *
 */
public class SplitMatrixOperation extends AbstractMatrixOperation {

    /**
     * Constructor for split matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     */
    public SplitMatrixOperation(int rows, int columns, int depth) {
        super(rows, columns, depth, true);
    }

    /**
     * Applies matrix operation.
     *
     * @param first first matrix.
     * @param splitAt splits at position
     * @param splitVertically if true splits vertically otherwise horizontally.
     * @return split matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix first, int splitAt, boolean splitVertically) throws MatrixException {
        if (!((first instanceof DMatrix) || (first instanceof SMatrix))) throw new MatrixException("Matrix must be of type DMatrix or SMatrix");
        Matrix matrix1;
        Matrix matrix2;
        int rows = getRows();
        int columns = getColumns();
        int totalDepth = getDepth();
        if (splitVertically) {
            if (splitAt < 1 || splitAt > rows - 1) throw new MatrixException("For vertical split position is beyond number of rows in matrix.");
            matrix1 = first.getNewMatrix(splitAt, columns, totalDepth);
            matrix2 = first.getNewMatrix(rows - matrix1.getTotalRows(), columns, totalDepth);
            for (int depth = 0; depth < totalDepth; depth++) {
                for (int row = 0; row < rows; row++) {
                    for (int column = 0; column < columns; column++) {
                        if (row < splitAt) matrix1.setValue(row, column, depth, first.getValue(row, column, depth));
                        else matrix2.setValue(row - splitAt, column, depth, first.getValue(row, column, depth));
                    }
                }
            }
        }
        else {
            if (splitAt < 1 || splitAt > columns - 1) throw new MatrixException("For vertical split position is beyond number of rows in matrix.");
            matrix1 = first.getNewMatrix(rows, splitAt, totalDepth);
            matrix2 = first.getNewMatrix(rows, columns - matrix1.getTotalColumns(), totalDepth);
            for (int depth = 0; depth < totalDepth; depth++) {
                for (int row = 0; row < rows; row++) {
                    for (int column = 0; column < columns; column++) {
                        if (column < splitAt) matrix1.setValue(row, column, depth, first.getValue(row, column, depth));
                        else matrix2.setValue(row, column - splitAt, depth, first.getValue(row, column, depth));
                    }
                }
            }
        }
        ArrayList<Matrix> matrices = new ArrayList<>();
        matrices.add(matrix1);
        matrices.add(matrix2);

        return new JMatrix(matrices, splitVertically);
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
