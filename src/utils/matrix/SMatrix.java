/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.matrix;

import java.util.HashMap;

/**
 * Matrix class that implements sparse matrix.<br>
 * Sparse matrix optimizes matrix memory usage by storing only non-zero values.<br>
 * This matrix type is useful when input sample is expected to contain mostly zero values.<br>
 *
 */
public class SMatrix extends ComputableMatrix {

    /**
     * Matrix data structure as hash map.
     *
     */
    private HashMap<Integer, Double> matrix = new HashMap<>();

    /**
     * Defines number of rows in matrix.
     *
     */
    private int rows;

    /**
     * Defines number of columns in matrix.
     *
     */
    private int columns;

    /**
     * Slice start row.
     *
     */
    private int sliceStartRow;

    /**
     * Slice start column.
     *
     */
    private int sliceStartColumn;

    /**
     * Number of rows in slice.
     *
     */
    private int sliceRows;

    /**
     * Number of columns in slice.
     *
     */
    private int sliceColumns;

    /**
     * Size of slice (sliceRows * sliceColumns)
     *
     */
    private int sliceSize;

    /**
     * Constructor for scalar matrix (size 1x1).
     *
     * @param scalarValue value for matrix.
     */
    public SMatrix(double scalarValue) {
        super(true);
        this.rows = 1;
        this.columns = 1;
        updateSliceDimensions(0, 0, 0, 0);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     */
    public SMatrix(int rows, int columns) {
        super(false);
        this.rows = rows;
        this.columns = columns;
        updateSliceDimensions(0, 0, rows - 1, columns - 1);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     */
    public SMatrix(int rows, int columns, Initialization initialization, int inputs, int outputs) {
        super(false);
        this.rows = rows;
        this.columns = columns;
        updateSliceDimensions(0, 0, rows - 1, columns - 1);
        initialize(initialization, inputs, outputs);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     */
    public SMatrix(int rows, int columns, Initialization initialization) {
        super(false);
        this.rows = rows;
        this.columns = columns;
        updateSliceDimensions(0, 0, rows - 1, columns - 1);
        initialize(initialization);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initializer initializer.
     */
    public SMatrix(int rows, int columns, Matrix.Initializer initializer) {
        super(false);
        this.rows = rows;
        this.columns = columns;
        updateSliceDimensions(0, 0, rows - 1, columns - 1);
        initialize(initializer);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param data clones matrix data from given matrix data.
     */
    public SMatrix(int rows, int columns, HashMap<Integer, Double> data) {
        super(false);
        this.rows = rows;
        this.columns = columns;
        updateSliceDimensions(0, 0, rows - 1, columns - 1);
        for (Integer index : data.keySet()) matrix.put(index, data.get(index));
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param data matrix data.
     * @param referTo if true creates matrix with reference to given matrix data otherwise clones the data.
     */
    public SMatrix(int rows, int columns, HashMap<Integer, Double> data, boolean referTo) {
        super(false);
        this.rows = rows;
        this.columns = columns;
        updateSliceDimensions(0, 0, rows - 1, columns - 1);
        if (referTo) matrix.putAll(data);
        else for (Integer index : data.keySet()) matrix.put(index, data.get(index));
    }

    /**
     * Updates slice dimensions.
     *
     */
    private void updateSliceDimensions(int startRow, int startColumn, int endRow, int endColumn) {
        sliceStartRow = startRow;
        sliceStartColumn = startColumn;
        sliceRows = endRow - sliceStartRow + 1;
        sliceColumns = endColumn - sliceStartColumn + 1;
        sliceSize = sliceRows * sliceColumns;
    }

    /**
     * Slices matrix.
     *
     * @param startRow start row of slice.
     * @param startColumn start column of slice.
     * @param endRow  end row of slice.
     * @param endColumn  end column of slice.
     * @throws MatrixException throws exception if slicing fails.
     */
    public void sliceAt(int startRow, int startColumn, int endRow, int endColumn) throws MatrixException {
        if (startRow < 0 || startColumn < 0 || endRow > rows -1 || endColumn > columns - 1) {
            throw new MatrixException("Slice rows: " + startRow + " - " + endRow + " and slice columns: " + startColumn + " - " + endColumn + " do not match matrix dimensions: " + rows + "x" + columns);
        }
        else updateSliceDimensions(startRow, startColumn, endRow, endColumn);
    }

    /**
     * Removes slicing of matrix.
     *
     */
    public void unslice() {
        updateSliceDimensions(0, 0, rows - 1, columns - 1);
    }

    /**
     * Resets matrix leaving dimensions same.
     *
     */
    public void resetMatrix() {
        matrix = new HashMap<>();
    }

    /**
     * Returns new mask for this matrix.
     *
     * @return mask of this matrix.
     */
    protected Mask getNewMask() {
        return new SMask(rows, columns);
    }

    /**
     * Sets value of matrix at specific row and column.
     *
     * @param row row of value to be set.
     * @param column column of value to be set.
     * @param value new value to be set.
     */
    public void setValue(int row, int column, double value) {
        if (value != 0) matrix.put(isScalar() ? 0 : (sliceStartRow + row) * columns + (sliceStartColumn + column), value);
    }

    /**
     * Returns value of matrix at specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @return value of row and column.
     */
    public double getValue(int row, int column) {
        return matrix.getOrDefault(isScalar() ? 0 : (sliceStartRow + row) * columns + (sliceStartColumn + column), (double)0);
    }

    /**
     * Returns size (rows * columns) of matrix
     *
     * @return size of matrix.
     */
    public int size() {
        return sliceSize;
    }

    /**
     * Returns number of rows in matrix.
     *
     * @return number of rows in matrix.
     */
    public int getRows() {
        return sliceRows;
    }

    /**
     * Returns number of columns in matrix.
     *
     * @return number of columns in matrix.
     */
    public int getColumns() {
        return sliceColumns;
    }

    /**
     * Returns new matrix of same dimensions.
     *
     * @return new matrix of same dimensions.
     */
    public Matrix getNewMatrix() {
        return isScalar() ? new SMatrix(0) : new SMatrix(getRows(), getColumns());
    }

    /**
     * Returns new matrix of same dimensions optionally as transposed.
     *
     * @param asTransposed if true returns new matrix as transposed otherwise with unchanged dimensions.
     * @return new matrix of same dimensions.
     */
    public Matrix getNewMatrix(boolean asTransposed) {
        return isScalar() ? new SMatrix(0) : !asTransposed ? new SMatrix(getRows(), getColumns()) :  new SMatrix(getColumns(), getRows());
    }

    /**
     * Copies new matrix data into this matrix. Assumes equal dimensions for both matrices.
     *
     * @param newMatrix new matrix to be copied inside this matrix.
     */
    public void copyMatrixData(Matrix newMatrix) {
        rows = newMatrix.getRows();
        columns = newMatrix.getColumns();
        matrix = new HashMap<>();
        for (int row = 0; row < newMatrix.getRows(); row++) {
            for (int column = 0; column < newMatrix.getColumns(); column++) {
                setValue(row, column, newMatrix.getValue(row, column));
            }
        }
    }

}
