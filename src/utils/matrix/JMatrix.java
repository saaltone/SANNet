package utils.matrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.TreeMap;

/**
 * Implements joined matrix which consists of horizontally or vertically concatenated DMatrix
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
     * Constructor for JMatrix.
     *
     * @param rows total number of rows.
     * @param columns total number of columns.
     * @param matrices matrices contained by JMatrix.
     * @param joinedVertically true if matrices are joined vertically otherwise matrices are joined horizontally.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public JMatrix(int rows, int columns, Matrix[] matrices, boolean joinedVertically) throws MatrixException {
        this(rows, columns, new ArrayList<>(Arrays.asList(matrices)), joinedVertically);
    }

    /**
     * Constructor for JMatrix.
     *
     * @param rows total number of rows.
     * @param columns total number of columns.
     * @param matrices matrices contained by JMatrix.
     * @param joinedVertically true if matrices are joined vertically otherwise matrices are joined horizontally.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public JMatrix(int rows, int columns, ArrayList<Matrix> matrices, boolean joinedVertically) throws MatrixException {
        super(rows, columns, false);
        this.matrices.addAll(matrices);
        this.joinedVertically = joinedVertically;
        if (rows != getRowsCount()) throw new MatrixException("Number of rows is not matching number of rows in assigned matrices.");
        if (columns != getColumnsCount()) throw new MatrixException("Number of columns is not matching number of columns in assigned matrices.");
        for (Matrix matrix : matrices) {
            if (matrix.isScalar()) throw new MatrixException("All matrices need to be non-scalar.");
        }
        updateSliceDimensions(0, 0, getTotalRows() - 1, getTotalColumns() - 1);
    }

    /**
     * Returns number of rows for JMatrix.
     *
     * @return number of rows for JMatrix.
     * @throws MatrixException throws matrix exception if matrix dimensions are not matching.
     */
    private int getRowsCount() throws MatrixException {
        int totalRows = 0;
        if (joinedVertically) {
            int columns = -1;
            for (Matrix matrix : matrices) {
                if (columns == -1) columns = matrix.getTotalColumns();
                else if (columns != matrix.getTotalColumns()) throw new MatrixException("Number of columns in matrices are not matching.");
                matrixPositionOffsets.put(totalRows, matrix);
                totalRows += matrix.getTotalRows();
            }
        } else totalRows = matrices.get(0).getTotalRows();
        return totalRows;
    }

    /**
     * Returns number of columns for JMatrix.
     *
     * @return number of columns for JMatrix.
     * @throws MatrixException throws matrix exception if matrix dimensions are not matching.
     */
    private int getColumnsCount() throws MatrixException {
        int totalColumns = 0;
        if (!joinedVertically) {
            int rows = -1;
            for (Matrix matrix : matrices) {
                if (rows == -1) rows = matrix.getTotalRows();
                else if (rows != matrix.getTotalRows()) throw new MatrixException("Number of rows in matrices are not matching.");
                matrixPositionOffsets.put(totalColumns, matrix);
                totalColumns += matrix.getTotalColumns();
            }
        } else totalColumns = matrices.get(0).getTotalColumns();
        return totalColumns;
    }

    /**
     * Returns sub-matrices within Matrix.
     *
     * @return sub-matrices within Matrix.
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
     * Returns new matrix of same dimensions.
     *
     * @return new matrix of same dimensions.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     *
     */
    public Matrix getNewMatrix() throws MatrixException {
        ArrayList<Matrix> newMatrices = new ArrayList<>();
        for (Matrix matrix : matrices) newMatrices.add(matrix.getNewMatrix());
        return new JMatrix(getTotalRows(), getTotalColumns(), newMatrices, joinedVertically);
    }

    /**
     * Returns new matrix of same dimensions optionally as transposed.
     *
     * @param asTransposed if true returns new matrix as transposed otherwise with unchanged dimensions.
     * @return new matrix of same dimensions.
     */
    public Matrix getNewMatrix(boolean asTransposed) {
        return isScalar() ? new DMatrix(0) : !asTransposed ? new DMatrix(getRows(), getColumns()) :  new DMatrix(getColumns(), getRows());
    }


}
