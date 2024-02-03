/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements dot operation.
 *
 */
public class DotMatrixOperation extends AbstractMatrixOperation {

    /**
     * First matrix.
     *
     */
    private transient Matrix first;

    /**
     * Second matrix.
     *
     */
    private transient Matrix second;

    /**
     * Number of rows in second matrix.
     *
     */
    private final int secondRows;

    /**
     * Constructor for dot matrix operation.
     *
     * @param firstRows  number of first matrix rows for operation.
     * @param secondRows number of second matrix rows for operation.
     * @param columns    number of columns for operation.
     * @param depth      depth for operation.
     */
    public DotMatrixOperation(int firstRows, int secondRows, int columns, int depth) {
        super(firstRows, columns, depth, false);
        this.secondRows = secondRows;
    }

    /**
     * Applies matrix operation.
     *
     * @param first  first matrix.
     * @param second second matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if new mask dimensions or mask type are not matching with this mask.
     */
    public Matrix apply(Matrix first, Matrix second) throws MatrixException {
        this.first = first;
        this.second = second;
        if (first.getColumns() != second.getRows() || first.getDepth() != second.getDepth()) {
            throw new MatrixException("Incompatible matrix sizes: " + first.getRows() + "x" + first.getColumns() + "x" + first.getDepth() + " by " + second.getRows() + "x" + second.getColumns() + "x" + second.getDepth());
        }
        return applyMatrixOperation(first, second, first.getNewMatrix(first.getRows(), second.getColumns(), getDepth()));
    }

    /**
     * Check if first matrix and optionally second matrix are masked at specific row and column.
     *
     * @param row    row.
     * @param column column.
     * @param first  first matrix.
     * @param second second matrix.
     * @return returns true if first or second matrix are masked at specific row and column.
     */
    protected boolean hasMaskAt(int row, int column, int depth, Matrix first, Matrix second) {
        return false;
    }

    /**
     * Applies matrix operation.
     *
     * @param first  first matrix.
     * @param second second matrix.
     * @param result result matrix.
     * @return result matrix.
     */
    protected Matrix applyMatrixOperation(Matrix first, Matrix second, Matrix result) {
        if (!hasMask(first, second)) {
            for (int depth = 0; depth < getDepth(); depth++) {
                for (int firstRow = 0; firstRow < getRows(); firstRow += getStride()) {
                    for (int secondRow = 0; secondRow < secondRows; secondRow += getStride()) {
                        apply(firstRow, secondRow, depth, 0, result);
                    }
                }
            }
        }
        else {
            for (int depth = 0; depth < getDepth(); depth++) {
                for (int firstRow = 0; firstRow < getRows(); firstRow += getStride()) {
                    for (int secondRow = 0; secondRow < secondRows; secondRow += getStride()) {
                        if (!hasMaskAt(firstRow, secondRow, depth, first, second)) {
                            applyMask(firstRow, secondRow, depth, 0, result);
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * Applies operation.
     *
     * @param firstRow  current firstRow.
     * @param secondRow current secondRow.
     * @param depth     current depth.
     * @param value     current value.
     * @param result    result matrix.
     */
    public void apply(int firstRow, int secondRow, int depth, double value, Matrix result) {
        for (int column = 0; column < getColumns(); column += getStride()) {
            result.setValue(firstRow, column, depth, result.getValue(firstRow, column, depth) + first.getValue(firstRow, secondRow, depth) * second.getValue(secondRow, column, depth));
        }
    }

    /**
     * Applies operation assuming masked matrices.
     *
     * @param firstRow  current firstRow.
     * @param secondRow current secondRow.
     * @param depth     current depth.
     * @param value     current value.
     * @param result    result matrix.
     */
    public void applyMask(int firstRow, int secondRow, int depth, double value, Matrix result) {
        for (int column = 0; column < getColumns(); column += getStride()) {
            if (!hasMaskAt(firstRow, secondRow, depth, first) && !hasMaskAt(secondRow, column, depth, second)) {
                result.setValue(firstRow, column, depth, result.getValue(firstRow, column, depth) + first.getValue(firstRow, secondRow, depth) * second.getValue(secondRow, column, depth));
            }
        }
    }

}
