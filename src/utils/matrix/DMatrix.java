/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.matrix;

/**
 * Matrix class that implements dense matrix.<br>
 * Dense matrix assumes full array data structure including storage of zero values.<br>
 *
 */
public class DMatrix extends Matrix {

    /**
     * Matrix data structure as two dimensional row column array.
     *
     */
    private double[][] matrix;

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param cols defines number of columns in matrix.
     */
    public DMatrix(int rows, int cols) {
        matrix = new double[rows][cols];
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param cols defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public DMatrix(int rows, int cols, Init initialization, int inputs, int outputs) throws MatrixException {
        matrix = new double[rows][cols];
        init(initialization, inputs, outputs);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param cols defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public DMatrix(int rows, int cols, Init initialization) throws MatrixException {
        matrix = new double[rows][cols];
        init(initialization);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param data clones matrix data from given matrix data.
     */
    public DMatrix(double[][] data) {
        matrix = data.clone();
    }

    /**
     * Constructor for dense matrix.
     *
     * @param data matrix data.
     * @param referTo if true creates matrix with reference to given matrix data otherwise clones the data.
     */
    public DMatrix(double[][] data, boolean referTo) {
        if (referTo) matrix = data;
        else matrix = data.clone();
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
     * Matrix function used to set value of specific row and column.
     *
     * @param row row of value to be set.
     * @param col column of value to be set.
     * @param value new value to be set.
     */
    public void setValue(int row, int col, double value) {
        matrix[!t ? row : col][!t ? col : row] = value;
    }

    /**
     * Matrix function used to get value of specific row and column.
     *
     * @param row row of value to be returned.
     * @param col column of value to be returned.
     * @return value of row and column.
     */
    public double getValue(int row, int col) {
        return matrix[!t ? row : col][!t ? col : row];
    }

    /**
     * Returns size (rows * columns) of matrix
     *
     * @return size of matrix.
     */
    public int getSize() {
        return matrix.length * matrix[0].length;
    }

    /**
     * Returns number of rows in matrix.
     *
     * @return number of rows in matrix.
     */
    public int getRows() {
        return !t ? matrix.length : matrix[0].length;
    }

    /**
     * Returns number of columns in matrix.
     *
     * @return number of columns in matrix.
     */
    public int getCols() {
        return !t ? matrix[0].length : matrix.length;
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
        matrix = new double[newMatrix.getRows()][newMatrix.getCols()];
        for (int row = 0; row < newMatrix.getRows(); row++) {
            for (int col = 0; col < newMatrix.getCols(); col++) {
                setValue(row, col, newMatrix.getValue(row, col));
            }
        }
    }

}
