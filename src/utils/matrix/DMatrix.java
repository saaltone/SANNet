/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.matrix;

import java.util.ArrayList;

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
     * Constructor for scalar matrix (size 1x1).
     *
     * @param scalarValue value for matrix.
     */
    public DMatrix(double scalarValue) {
        super(1, 1,true);
        matrix = new double[getTotalRows()][getTotalColumns()];
        matrix[0][0] = scalarValue;
        updateSliceDimensions(0, 0, 0, 0);
    }

    /**
     * Constructor for scalar matrix (size 1x1).
     *
     * @param scalarValue value for matrix.
     * @param name name of matrix.
     */
    public DMatrix(double scalarValue, String name) {
        super(1, 1,true, name);
        matrix = new double[getTotalRows()][getTotalColumns()];
        matrix[getTotalRows() - 1][getTotalColumns() - 1] = scalarValue;
        updateSliceDimensions(0, 0, 0, 0);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     */
    public DMatrix(int rows, int columns) {
        super(rows, columns, false);
        matrix = new double[rows][columns];
        updateSliceDimensions(0, 0, rows - 1, columns - 1);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param name name of matrix.
     */
    public DMatrix(int rows, int columns, String name) {
        super(rows, columns, false, name);
        matrix = new double[rows][columns];
        updateSliceDimensions(0, 0, rows - 1, columns - 1);
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
        this(rows, columns);
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
        this(rows, columns, name);
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
        this(rows, columns);
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
        this(rows, columns, name);
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
        this(rows, columns, name);
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
        this(rows, columns);
        initialize(initializer);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param data clones matrix data from given matrix data.
     */
    public DMatrix(double[][] data) {
        super(data.length, data[0].length, false);
        matrix = data.clone();
        updateSliceDimensions(0, 0, getTotalRows() - 1, getTotalColumns() - 1);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param data matrix data.
     * @param referTo if true creates matrix with reference to given matrix data otherwise clones the data.
     */
    public DMatrix(double[][] data, boolean referTo) {
        super(data.length, data[0].length, false);
        matrix = referTo ? data : data.clone();
        updateSliceDimensions(0, 0, getTotalRows() - 1, getTotalColumns() - 1);
    }

    /**
     * Returns sub-matrices within Matrix.
     *
     * @return sub-matrices within Matrix.
     */
    public ArrayList<Matrix> getSubMatrices() {
        ArrayList<Matrix> matrices = new ArrayList<>();
        matrices.add(this);
        return matrices;
    }

    /**
     * Resets matrix leaving dimensions same.
     *
     */
    public void resetMatrix() {
        matrix = new double[getTotalRows()][getTotalColumns()];
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
        matrix[isScalar() ? 0 : getSliceStartRow() + row][isScalar() ? 0 : getSliceStartColumn() + column] = value;
    }

    /**
     * Returns value of matrix at specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @return value of row and column.
     */
    public double getValue(int row, int column) {
        return matrix[isScalar() ? 0 : getSliceStartRow() + row][isScalar() ? 0 : getSliceStartColumn() + column];
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

}
