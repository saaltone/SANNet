/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
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
    public JMatrix(TreeMap<Integer, Matrix> matrices, boolean joinedVertically) throws MatrixException {
        this(new ArrayList<>(matrices.values()), joinedVertically);
    }

    /**
     * Constructor for joined matrix.
     *
     * @param matrices matrices contained by joined matrix.
     * @param joinedVertically true if matrices are joined vertically otherwise matrices are joined horizontally.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public JMatrix(ArrayList<Matrix> matrices, boolean joinedVertically) throws MatrixException {
        super(joinedVertically ? matrices.stream().mapToInt(Matrix::getTotalRows).sum() : matrices.get(0).getTotalRows(), joinedVertically ? matrices.get(0).getTotalColumns() : matrices.stream().mapToInt(Matrix::getTotalColumns).sum(), matrices.get(0).getTotalDepth(), false);

        this.matrices.addAll(matrices);
        this.joinedVertically = joinedVertically;

        if (joinedVertically) {
            int totalRows = 0;
            int columns = -1;
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
     * Creates new matrix with object full copy of this matrix.
     *
     * @return newly created reference matrix.
     * @throws MatrixException throws exception if mask is not set or cloning of matrix fails.
     */
    public Matrix copy(boolean canBeSliced) throws MatrixException {
        ArrayList<Matrix> subMatrices = new ArrayList<>();
        for (Matrix matrix : matrices) subMatrices.add(matrix.copy());
        return new JMatrix(subMatrices, joinedVertically);
    }

    /**
     * Redimensions matrix assuming new dimensions are matching.
     *
     * @param newRows new row size
     * @param newColumns new column size
     * @param newDepth new depth size.
     * @return redimensioned matrix.
     * @throws MatrixException throws exception if redimensioning fails.
     */
    public Matrix redimension(int newRows, int newColumns, int newDepth) throws MatrixException {
        throw new MatrixException("JMatrix type cannot be redimensioned.");
    }

    /**
     * Redimensions matrix assuming new dimensions are matching.
     *
     * @param newRows new row size
     * @param newColumns new column size
     * @param newDepth new depth size.
     * @param copyData if true matrix data is copied and if false referenced.
     * @return redimensioned matrix.
     * @throws MatrixException throws exception if redimensioning fails.
     */
    public Matrix redimension(int newRows, int newColumns, int newDepth, boolean copyData) throws MatrixException {
        throw new MatrixException("JMatrix type cannot be redimensioned.");
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
    public Matrix applyTranspose() throws MatrixException {
        ArrayList<Matrix> transposedSubMatrices = new ArrayList<>();
        for (Matrix subMatrix : getSubMatrices()) transposedSubMatrices.add(subMatrix.transpose());
        return new JMatrix(transposedSubMatrices, joinedVertically == isTransposed());
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
        return new DMask(getTotalRows(), getTotalColumns(), getTotalDepth());
    }

    /**
     * Sets value of matrix at specific row and column.
     *
     * @param row row of value to be set.
     * @param column column of value to be set.
     * @param depth depth of value to be set.
     * @param value new value to be set.
     */
    public void setValue(int row, int column, int depth, double value) {
        int realRow = getSliceStartRow() + row;
        int realColumn = getSliceStartColumn() + column;
        if (joinedVertically) {
            int startRow = matrixPositionOffsets.floorKey(realRow);
            Matrix matrix = matrixPositionOffsets.get(startRow);
            matrix.setValue(realRow - startRow, realColumn, depth, value);
        }
        else {
            int startColumn = matrixPositionOffsets.floorKey(realColumn);
            Matrix matrix = matrixPositionOffsets.get(startColumn);
            matrix.setValue(realRow, realColumn - startColumn, depth, value);
        }
    }

    /**
     * Returns value of matrix at specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @param depth depth of value to be returned.
     * @return value of row and column.
     */
    public double getValue(int row, int column, int depth) {
        int realRow = getSliceStartRow() + row;
        int realColumn = getSliceStartColumn() + column;
        if (joinedVertically) {
            int startRow = matrixPositionOffsets.floorKey(realRow);
            Matrix matrix = matrixPositionOffsets.get(startRow);
            return matrix.getValue(realRow - startRow, realColumn, depth);
        }
        else {
            int startColumn = matrixPositionOffsets.floorKey(realColumn);
            Matrix matrix = matrixPositionOffsets.get(startColumn);
            return matrix.getValue(realRow, realColumn - startColumn, depth);
        }
    }

    /**
     * Returns matrix of given size (rows x columns x depth)
     *
     * @param rows rows
     * @param columns columns
     * @param depth depth
     * @return new matrix
     */
    public Matrix getNewMatrix(int rows, int columns, int depth) {
        return new DMatrix(rows, columns, depth);
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
