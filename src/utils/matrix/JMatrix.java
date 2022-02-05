/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.TreeMap;

/**
 * Implements joined matrix which consists of horizontally or vertically concatenated matrices
 *
 */
public class JMatrix extends ComputableMatrix {

    /**
     * Array list of Matrices forming JMatrix.
     *
     */
    private final ArrayList<Matrix> matrices = new ArrayList<>();

    /**
     * Hash map of matrix position offsets.
     *
     */
    private final TreeMap<Integer, Matrix> matrixPositionOffsets = new TreeMap<>();

    /**
     * True when matrices are joined vertically otherwise indicates that matrices are joined horizontally.
     *
     */
    private final boolean joinedVertically;

    /**
     * Constructor for joined matrix.
     *
     * @param matrices matrices contained by joined matrix.
     * @param joinedVertically true if matrices are joined vertically otherwise matrices are joined horizontally.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public JMatrix(Matrix[] matrices, boolean joinedVertically) throws MatrixException {
        this(new ArrayList<>(Arrays.asList(matrices)), joinedVertically);
    }

    /**
     * Constructor for joined matrix.
     *
     * @param matrices matrices contained by joined matrix.
     * @param joinedVertically true if matrices are joined vertically otherwise matrices are joined horizontally.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public JMatrix(ArrayList<Matrix> matrices, boolean joinedVertically) throws MatrixException {
        super(joinedVertically ? matrices.stream().mapToInt(Matrix::getTotalRows).sum() : matrices.get(0).getTotalRows(), joinedVertically ? matrices.get(0).getTotalColumns() : matrices.stream().mapToInt(Matrix::getTotalColumns).sum(), false);

        this.matrices.addAll(matrices);
        this.joinedVertically = joinedVertically;

        if (joinedVertically) {
            int columns = -1;
            int totalRows = 0;
            for (Matrix matrix : matrices) {
                int matrixTotalColumns = matrix.getTotalColumns();
                if (columns == -1) columns = matrixTotalColumns;
                else if (columns != matrixTotalColumns) throw new MatrixException("Number of columns in matrices are not matching.");
                matrixPositionOffsets.put(totalRows, matrix);
                totalRows += matrix.getTotalRows();
            }
        }
        else {
            int rows = -1;
            int totalColumns = 0;
            for (Matrix matrix : matrices) {
                int matrixTotalRows = matrix.getTotalRows();
                if (rows == -1) rows = matrixTotalRows;
                else if (rows != matrixTotalRows) throw new MatrixException("Number of rows in matrices are not matching.");
                matrixPositionOffsets.put(totalColumns, matrix);
                totalColumns += matrix.getTotalColumns();
            }
        }

        updateSliceDimensions(0, 0, getTotalRows() - 1, getTotalColumns() - 1);
    }

    /**
     * Creates new matrix with object full copy of this matrix.
     *
     * @return newly created reference matrix.
     * @throws MatrixException throws exception if mask is not set or cloning of matrix fails.
     */
    public Matrix copy() throws MatrixException {
        ArrayList<Matrix> subMatrices = new ArrayList<>();
        for (Matrix matrix : matrices) subMatrices.add(matrix.copy());
        return new JMatrix(subMatrices, joinedVertically);
    }

    /**
     * Checks if data of other matrix is equal to data of this matrix
     *
     * @param other matrix to be compared.
     * @return true is data of this and other matrix are equal otherwise false.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public boolean equals(Matrix other) throws MatrixException {
        if (other instanceof JMatrix otherJMatrix) {
            if (other.getRows() != getRows() || other.getColumns() != getColumns()) {
                throw new MatrixException("Incompatible target matrix size: " + other.getRows() + "x" + other.getColumns());
            }
            ArrayList<Matrix> otherSubMatrices = otherJMatrix.getSubMatrices();
            for (int index = 0; index < matrices.size(); index++) if (!matrices.get(index).equals(otherSubMatrices.get(index))) return false;
            return true;
        }
        else return super.equals(other);
    }

    /**
     * Transposes matrix.
     *
     * @return transposed matrix.
     * @throws MatrixException throws exception if cloning of mask fails.
     */
    public Matrix transpose() throws MatrixException {
        ArrayList<Matrix> transposedSubMatrices = new ArrayList<>();
        ArrayList<Matrix> subMatrices = getSubMatrices();
        for (Matrix matrix : subMatrices) transposedSubMatrices.add(matrix.transpose());
        return new JMatrix(transposedSubMatrices, !joinedVertically);
    }

    /**
     * Returns sub-matrices within matrix.
     *
     * @return sub-matrices within matrix.
     */
    public ArrayList<Matrix> getSubMatrices() {
        return new ArrayList<>(matrices);
    }

    /**
     * Resets matrix leaving dimensions same.
     *
     */
    public void resetMatrix() {
        for (Matrix matrix : matrices) matrix.reset();
    }

    /**
     * Returns new mask for this matrix.
     *
     * @return mask of this matrix.
     */
    protected Mask getNewMask() {
        return new DMask(getTotalRows(), getTotalColumns());
    }

    /**
     * Sets value of matrix at specific row and column.
     *
     * @param row row of value to be set.
     * @param column column of value to be set.
     * @param value new value to be set.
     */
    public void setValue(int row, int column, double value) {
        int realRow = getSliceStartRow() + row;
        int realColumn = getSliceStartColumn() + column;
        if (joinedVertically) {
            int startRow = matrixPositionOffsets.floorKey(realRow);
            Matrix matrix = matrixPositionOffsets.get(startRow);
            matrix.setValue(realRow - startRow, realColumn, value);
        }
        else {
            int startColumn = matrixPositionOffsets.floorKey(realColumn);
            Matrix matrix = matrixPositionOffsets.get(startColumn);
            matrix.setValue(realRow, realColumn - startColumn, value);
        }
    }

    /**
     * Returns value of matrix at specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @return value of row and column.
     */
    public double getValue(int row, int column) {
        int realRow = getSliceStartRow() + row;
        int realColumn = getSliceStartColumn() + column;
        if (joinedVertically) {
            int startRow = matrixPositionOffsets.floorKey(realRow);
            Matrix matrix = matrixPositionOffsets.get(startRow);
            return matrix.getValue(realRow - startRow, realColumn);
        }
        else {
            int startColumn = matrixPositionOffsets.floorKey(realColumn);
            Matrix matrix = matrixPositionOffsets.get(startColumn);
            return matrix.getValue(realRow, realColumn - startColumn);
        }
    }

    /**
     * Returns matrix of given size (rows x columns)
     *
     * @param rows rows
     * @param columns columns
     * @return new matrix
     */
    protected Matrix getNewMatrix(int rows, int columns) {
        return new DMatrix(rows, columns);
    }

    /**
     * Returns constant matrix
     *
     * @param constant constant
     * @return new matrix
     */
    protected Matrix getNewMatrix(double constant) {
        return new DMatrix(constant);
    }

    /**
     * Returns new matrix of same dimensions.
     *
     * @return new matrix of same dimensions.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     *
     */
    public Matrix getNewMatrix() throws MatrixException {
        ArrayList<Matrix> newMatrices = new ArrayList<>();
        for (Matrix matrix : matrices) newMatrices.add(matrix.getNewMatrix());
        return new JMatrix(newMatrices, joinedVertically);
    }


}
