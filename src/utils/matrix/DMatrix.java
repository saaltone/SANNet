/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.matrix;

/**
 * Matrix class that implements dense matrix.<br>
 * Dense matrix assumes full array data structure including storage of zero values.<br>
 *
 */
public class DMatrix extends ComputableMatrix {

    /**
     * Matrix data structure as two dimensional row column array.
     *
     */
    private double[][] matrix;

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
    public DMatrix(double scalarValue) {
        super(true);
        matrix = new double[1][1];
        matrix[0][0] = scalarValue;
        updateSliceDimensions(0, 0, matrix.length - 1, matrix[0].length - 1);
    }

    /**
     * Constructor for scalar matrix (size 1x1).
     *
     * @param scalarValue value for matrix.
     * @param name name of matrix.
     */
    public DMatrix(double scalarValue, String name) {
        super(true, name);
        matrix = new double[1][1];
        matrix[0][0] = scalarValue;
        updateSliceDimensions(0, 0, matrix.length - 1, matrix[0].length - 1);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     */
    public DMatrix(int rows, int columns) {
        super(false);
        matrix = new double[rows][columns];
        updateSliceDimensions(0, 0, matrix.length - 1, matrix[0].length - 1);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param name name of matrix.
     */
    public DMatrix(int rows, int columns, String name) {
        super(false, name);
        matrix = new double[rows][columns];
        updateSliceDimensions(0, 0, matrix.length - 1, matrix[0].length - 1);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     */
    public DMatrix(int rows, int columns, Initialization initialization, int inputs, int outputs) {
        super(false);
        matrix = new double[rows][columns];
        updateSliceDimensions(0, 0, matrix.length - 1, matrix[0].length - 1);
        initialize(initialization, inputs, outputs);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     * @param name name of matrix.
     */
    public DMatrix(int rows, int columns, Initialization initialization, int inputs, int outputs, String name) {
        super(false, name);
        matrix = new double[rows][columns];
        updateSliceDimensions(0, 0, matrix.length - 1, matrix[0].length - 1);
        initialize(initialization, inputs, outputs);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     */
    public DMatrix(int rows, int columns, Initialization initialization) {
        super(false);
        matrix = new double[rows][columns];
        updateSliceDimensions(0, 0, matrix.length - 1, matrix[0].length - 1);
        initialize(initialization);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     * @param name name of matrix.
     */
    public DMatrix(int rows, int columns, Initialization initialization, String name) {
        super(false, name);
        matrix = new double[rows][columns];
        updateSliceDimensions(0, 0, matrix.length - 1, matrix[0].length - 1);
        initialize(initialization);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initializer initializer.
     * @param name name of matrix.
     */
    public DMatrix(int rows, int columns, Matrix.Initializer initializer, String name) {
        super(false, name);
        matrix = new double[rows][columns];
        updateSliceDimensions(0, 0, matrix.length - 1, matrix[0].length - 1);
        initialize(initializer);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initializer initializer.
     */
    public DMatrix(int rows, int columns, Matrix.Initializer initializer) {
        super(false);
        matrix = new double[rows][columns];
        updateSliceDimensions(0, 0, matrix.length - 1, matrix[0].length - 1);
        initialize(initializer);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param data clones matrix data from given matrix data.
     */
    public DMatrix(double[][] data) {
        super(false);
        matrix = data.clone();
        updateSliceDimensions(0, 0, matrix.length - 1, matrix[0].length - 1);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param data matrix data.
     * @param referTo if true creates matrix with reference to given matrix data otherwise clones the data.
     */
    public DMatrix(double[][] data, boolean referTo) {
        super(false);
        if (referTo) matrix = data;
        else matrix = data.clone();
        updateSliceDimensions(0, 0, matrix.length - 1, matrix[0].length - 1);
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
        if (startRow < 0 || startColumn < 0 || endRow > matrix.length -1 || endColumn > matrix[0].length - 1) {
            throw new MatrixException("Slice rows: " + startRow + " - " + endRow + " and slice columns: " + startColumn + " - " + endColumn + " do not match matrix dimensions: " + matrix.length + "x" + matrix[0].length);
        }
        else updateSliceDimensions(startRow, startColumn, endRow, endColumn);
    }

    /**
     * Removes slicing of matrix.
     *
     */
    public void unslice() {
        updateSliceDimensions(0, 0, matrix.length - 1, matrix[0].length - 1);
    }

    /**
     * Resets matrix leaving dimensions same.
     *
     */
    public void resetMatrix() {
        matrix = new double[matrix.length][matrix[0].length];
    }

    /**
     * Returns new mask for this matrix.
     *
     * @return mask of this matrix.
     */
    protected Mask getNewMask() {
        return new DMask(matrix.length, matrix[0].length);
    }

    /**
     * Sets value of matrix at specific row and column.
     *
     * @param row row of value to be set.
     * @param column column of value to be set.
     * @param value new value to be set.
     */
    public void setValue(int row, int column, double value) {
        matrix[isScalar() ? 0 : sliceStartRow + row][isScalar() ? 0 : sliceStartColumn + column] = value;
    }

    /**
     * Returns value of matrix at specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @return value of row and column.
     */
    public double getValue(int row, int column) {
        return matrix[isScalar() ? 0 : sliceStartRow + row][isScalar() ? 0 : sliceStartColumn + column];
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
        return isScalar() ? new DMatrix(0) : new DMatrix(getRows(), getColumns());
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

    /**
     * Copies new matrix data into this matrix. Assumes equal dimensions for both matrices.
     *
     * @param newMatrix new matrix to be copied inside this matrix.
     */
    public void copyMatrixData(Matrix newMatrix) {
        matrix = new double[newMatrix.getRows()][newMatrix.getColumns()];
        for (int row = 0; row < newMatrix.getRows(); row++) {
            for (int column = 0; column < newMatrix.getColumns(); column++) {
                setValue(row, column, newMatrix.getValue(row, column));
            }
        }
    }

}
