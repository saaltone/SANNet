/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package utils;

import java.util.Stack;

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
     * Matrix mask data structure as two dimensional row column array-
     *
     */
    private boolean[][] mask;

    /**
     * Matrix row mask data structure.
     *
     */
    private boolean[] rowMask;

    /**
     * Matrix column mask data structure.
     *
     */
    private boolean[] colMask;

    /**
     * Stack to store matrix masks.
     */
    private Stack<boolean[][]> maskStack = new Stack<>();

    /**
     * Stack to store matrix row masks.
     *
     */
    private Stack<boolean[]> rowMaskStack = new Stack<>();

    /**
     * Stack to store matrix column masks.
     *
     */
    private Stack<boolean[]> colMaskStack = new Stack<>();

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param cols defines number of columns in matrix.
     */
    public DMatrix(int rows, int cols) {
        matrix = new double[rows][cols];
        initializeSlice();
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param cols defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     */
    public DMatrix(int rows, int cols, Init initialization, int inputs, int outputs) {
        matrix = new double[rows][cols];
        init(initialization, inputs, outputs);
        initializeSlice();
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param cols defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     */
    public DMatrix(int rows, int cols, Init initialization) {
        matrix = new double[rows][cols];
        init(initialization);
        initializeSlice();
    }

    /**
     * Constructor for dense matrix.
     *
     * @param data clones matrix data from given matrix data.
     */
    public DMatrix(double[][] data) {
        matrix = data.clone();
        initializeSlice();
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
        initializeSlice();
    }

    /**
     * Resets matrix leaving dimensions same.
     *
     */
    public void resetMatrix() {
        matrix = new double[matrix.length][matrix[0].length];
        initializeSlice();
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
     * Gets size (rows * columns) of matrix
     *
     * @return size of matrix.
     */
    public int getSize() {
        return matrix.length * matrix[0].length;
    }

    /**
     * Gets number of rows in matrix.
     *
     * @return number of rows in matrix.
     */
    public int getRows() {
        return !t ? matrix.length : matrix[0].length;
    }

    /**
     * Gets number of columns in matrix.
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
    /**
     * Matrix internal function used to set matrix masking of specific row and column.
     *
     * @param row row of value to be get.
     * @param col column of value to be get.
     * @param value defines if specific matrix row and column is masked (true) or not (false).
     */
    protected void setMaskValue(int row, int col, boolean value) {
        mask[!t ? row : col][!t ? col : row] = value;
    }

    /**
     * Matrix internal function used to get matrix masking of specific row and column.
     *
     * @param row row of value to be returned.
     * @param col column of value to be returned.
     * @return if specific matrix row and column is masked (true) or not (false).
     */
    protected boolean getMaskValue(int row, int col) {
        return mask[!t ? row : col][!t ? col : row];
    }


    /**
     * Clears and removes mask from this matrix.
     *
     */
    public void noMask() {
        mask = null;
        rowMask = null;
        colMask = null;
        maskStack = new Stack<>();
        rowMaskStack = new Stack<>();
        colMaskStack = new Stack<>();
    }

    /**
     * Checks if mask is set.
     *
     * @throws MatrixException throws exception if mask is not set.
     */
    protected void checkMask() throws MatrixException {
        if (mask == null) throw new MatrixException("Mask is not set");
    }

    /**
     * Resets current mask.
     *
     */
    public void resetMask() {
        mask = new boolean[matrix.length][matrix[0].length];
        rowMask = new boolean[matrix.length];
        colMask = new boolean[matrix[0].length];
    }

    /**
     * Pushes current mask into stack and optionally creates new mask for this matrix.<br>
     * Useful in operations where sequence of operations are taken between this matrix and other matrices.<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     * @throws MatrixException throws exception if mask is not set.
     */
    public void stackMask(boolean reset) throws MatrixException {
        checkMask();
        maskStack.push(mask);
        if (reset) mask = new boolean[matrix.length][matrix[0].length];
    }

    /**
     * Pops mask from mask stack.
     *
     * @throws MatrixException throws exception if mask stack is empty.
     */
    public void unstackMask() throws MatrixException {
        if (maskStack.isEmpty()) throw new MatrixException("Mask stack is empty.");
        mask = maskStack.pop();
    }

    /**
     * Returns size of a mask stack.
     *
     * @return Size of mask stack.
     */
    public int maskStackSize() {
        return maskStack.size();
    }

    /**
     * Clears mask stack.
     *
     */
    public void clearMaskStack() {
        maskStack = new Stack<>();
    }

    /**
     * Checks if row mask is set.
     *
     * @throws MatrixException throws exception if row mask is not set.
     */
    protected void checkRowMask() throws MatrixException {
        if (rowMask == null) throw new MatrixException("Row mask is not set");
    }

    /**
     * Pushes current row mask into stack and optionally creates new mask for this matrix.<br>
     * Useful in operations where sequence of operations are taken between this matrix and other matrices.<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     * @throws MatrixException throws exception if mask is not set.
     */
    public void stackRowMask(boolean reset) throws MatrixException {
        checkRowMask();
        rowMaskStack.push(rowMask);
        if (reset) rowMask = new boolean[matrix.length];
    }

    /**
     * Pops row mask from mask stack.
     *
     * @throws MatrixException throws exception if row mask stack is empty.
     */
    public void unstackRowMask() throws MatrixException {
        if (rowMaskStack.isEmpty()) throw new MatrixException("Row mask stack is empty.");
        rowMask = rowMaskStack.pop();
    }

    /**
     * Returns size of a row mask stack.
     *
     * @return size of row mask stack.
     */
    public int rowMaskStackSize() {
        return rowMaskStack.size();
    }

    /**
     * Clears row mask stack.
     *
     */
    public void clearRowMaskStack() {
        rowMaskStack = new Stack<>();
    }

    /**
     * Sets mask value for row mask.
     *
     * @param row row of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     * @throws MatrixException throws exception if row is beyond dimension of current row mask.
     */
    protected void setRowMaskValue(int row, boolean value) throws MatrixException {
        checkRowMask();
        if (!t) rowMask[row] = value;
        else colMask[row] = value;
    }

    /**
     * Gets mask value for row mask.
     *
     * @param row row of mask to be returned.
     * @return true if row mask is set otherwise false.
     * @throws MatrixException throws exception if row is beyond dimension of current row mask or row mask is not set.
     */
    protected boolean getRowMaskValue(int row) throws MatrixException {
        checkRowMask();
        return !t ? rowMask[row] : colMask[row];
    }

    /**
     * Checks if column mask is set.
     *
     * @throws MatrixException throws exception if column mask is not set.
     */
    protected void checkColMask() throws MatrixException {
        if (colMask == null) throw new MatrixException("Column mask is not set");
    }

    /**
     * Sets mask value for column mask.
     *
     * @param col column of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     * @throws MatrixException throws exception if column is beyond dimension of current column mask or column mask is not set.
     */
    protected void setColMaskValue(int col, boolean value) throws MatrixException {
        checkColMask();
        if (!t) colMask[col] = value;
        else rowMask[col] = value;
    }

    /**
     * Gets mask value for column mask.
     *
     * @param col column of mask to be returned.
     * @return true if row mask is set otherwise false.
     * @throws MatrixException throws exception if column is beyond dimension of current column mask.
     */
    protected boolean getColMaskValue(int col) throws MatrixException {
        checkColMask();
        return !t ? colMask[col] : rowMask[col];
    }

    /**
     * Pushes current column mask into stack and optionally creates new mask for this matrix.<br>
     * Useful in operations where sequence of operations are taken between this matrix and other matrices.<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     * @throws MatrixException throws exception if mask is not set.
     */
    public void stackColMask(boolean reset) throws MatrixException {
        checkColMask();
        colMaskStack.push(colMask);
        if (reset) colMask = new boolean[matrix[0].length];
    }

    /**
     * Pops column mask from mask stack.
     *
     * @throws MatrixException throws exception if column mask stack is empty.
     */
    public void unstackColMask() throws MatrixException {
        if (colMaskStack.isEmpty()) throw new MatrixException("Column mask stack is empty.");
        colMask = colMaskStack.pop();
    }

    /**
     * Returns size of a column mask stack.
     *
     * @return size of column mask stack.
     */
    public int colMaskStackSize() {
        return colMaskStack.size();
    }

    /**
     * Clears column mask stack.
     *
     */
    public void clearColMaskStack() {
        colMaskStack = new Stack<>();
    }

}
