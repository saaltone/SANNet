/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.matrix;

import java.util.HashMap;

/**
 * Matrix class that implements sparse matrix.<br>
 * Sparse matrix optimizes matrix memory usage by storing only non-zero values.<br>
 * This matrix type is useful when input sample is expected to contain mostly zero values.<br>
 *
 */
public class SMatrix extends Matrix {

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
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     */
    public SMatrix(int rows, int columns) {
        super(false);
        this.rows = rows;
        this.columns = columns;
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
        initialize(initialization);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initializer initializer.
     */
    public SMatrix(int rows, int columns, Initializer initializer) {
        super(false);
        this.rows = rows;
        this.columns = columns;
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
        if (referTo) matrix.putAll(data);
        else for (Integer index : data.keySet()) matrix.put(index, data.get(index));
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
     * Matrix function used to set value of specific row and column.
     *
     * @param row row of value to be set.
     * @param column column of value to be set.
     * @param value new value to be set.
     */
    public void setValue(int row, int column, double value) {
        if (value != 0) matrix.put((!isTransposed ? row : column) * columns + (!isTransposed ? column : row), value);
    }

    /**
     * Matrix function used to get value of specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @return value of row and column.
     */
    public double getValue(int row, int column) {
        return matrix.getOrDefault((!isTransposed ? row : column) * columns + (!isTransposed ? column : row), (double)0);
    }

    /**
     * Returns size (rows * columns) of matrix
     *
     * @return size of matrix.
     */
    public int size() {
        return rows * columns;
    }

    /**
     * Returns number of rows in matrix.
     *
     * @return number of rows in matrix.
     */
    public int getRows() {
        return !isTransposed ? rows : columns;
    }

    /**
     * Returns number of columns in matrix.
     *
     * @return number of columns in matrix.
     */
    public int getColumns() {
        return !isTransposed ? columns : rows;
    }

    /**
     * Returns new matrix of dimensions rows x columns
     *
     * @return new matrix of dimensions rows x columns.
     */
    public Matrix getNewMatrix() {
        return new SMatrix(getRows(), getColumns());
    }

    /**
     * Copies new matrix into this matrix. Assumes equal dimensions for both matrices.
     *
     * @param newMatrix new matrix to be copied inside this matrix.
     */
    protected void copyMatrixData(Matrix newMatrix) {
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
