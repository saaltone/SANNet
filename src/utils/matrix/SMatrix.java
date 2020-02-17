/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.matrix;

import utils.matrix.*;

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
    private int cols;

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param cols defines number of columns in matrix.
     */
    public SMatrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param cols defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public SMatrix(int rows, int cols, Init initialization, int inputs, int outputs) throws MatrixException {
        this.rows = rows;
        this.cols = cols;
        init(initialization, inputs, outputs);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param cols defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public SMatrix(int rows, int cols, Init initialization) throws MatrixException {
        this.rows = rows;
        this.cols = cols;
        init(initialization);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param cols defines number of columns in matrix.
     * @param data clones matrix data from given matrix data.
     */
    public SMatrix(int rows, int cols, HashMap<Integer, Double> data) {
        this.rows = rows;
        this.cols = cols;
        for (Integer index : data.keySet()) matrix.put(index, data.get(index));
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param cols defines number of columns in matrix.
     * @param data matrix data.
     * @param referTo if true creates matrix with reference to given matrix data otherwise clones the data.
     */
    public SMatrix(int rows, int cols, HashMap<Integer, Double> data, boolean referTo) {
        this.rows = rows;
        this.cols = cols;
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
        return new SMask(rows, cols);
    }

    /**
     * Matrix function used to set value of specific row and column.
     *
     * @param row row of value to be set.
     * @param col column of value to be set.
     * @param value new value to be set.
     */
    public void setValue(int row, int col, double value) {
        if (value == 0) return;
        int curRow = !t ? row : col;
        int curCol = !t ? col : row;
        matrix.put(curRow * cols + curCol, value);
    }

    /**
     * Matrix function used to get value of specific row and column.
     *
     * @param row row of value to be returned.
     * @param col column of value to be returned.
     * @return value of row and column.
     */
    public double getValue(int row, int col) {
        int curRow = !t ? row : col;
        int curCol = !t ? col : row;
        return matrix.getOrDefault(curRow * cols + curCol, (double)0);
    }

    /**
     * Returns size (rows * columns) of matrix
     *
     * @return size of matrix.
     */
    public int getSize() {
        return rows * cols;
    }

    /**
     * Returns number of rows in matrix.
     *
     * @return number of rows in matrix.
     */
    public int getRows() {
        return !t ? rows : cols;
    }

    /**
     * Returns number of columns in matrix.
     *
     * @return number of columns in matrix.
     */
    public int getCols() {
        return !t ? cols : rows;
    }

    /**
     * Returns new matrix of dimensions rows x columns
     *
     * @param rows amount of rows for new matrix.
     * @param cols amount of columns for new matrix.
     * @return new matrix of dimensions rows x columns.
     */
    protected Matrix getNewMatrix(int rows, int cols) {
        return new DMatrix(rows, cols);
    }

    /**
     * Copies new matrix into this matrix. Assumes equal dimensions for both matrices.
     *
     * @param newMatrix new matrix to be copied inside this matrix.
     */
    protected void setAsMatrix(Matrix newMatrix) {
        rows = newMatrix.getRows();
        cols = newMatrix.getCols();
        matrix = new HashMap<>();
        for (int row = 0; row < newMatrix.getRows(); row++) {
            for (int col = 0; col < newMatrix.getCols(); col++) {
                setValue(row, col, newMatrix.getValue(row, col));
            }
        }
    }

}
