/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package utils;

import java.util.HashMap;
import java.util.Stack;

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
     * Hash map to store mask information.
     *
     */
    private HashMap<Integer, Boolean> mask = new HashMap<>();

    /**
     * Hash map to store row mask information.
     *
     */
    private HashMap<Integer, Boolean> rowMask = new HashMap<>();

    /**
     * Hash map to store column mask information.
     *
     */
    private HashMap<Integer, Boolean> colMask = new HashMap<>();

    /**
     * Defines if mask has been set for matrix.
     */
    private boolean hasMask = false;

    /**
     * Stack to store matrix masks.
     */
    private Stack<HashMap<Integer, Boolean>> maskStack = new Stack<>();

    /**
     * Stack to store matrix row masks.
     *
     */
    private Stack<HashMap<Integer, Boolean>> rowMaskStack = new Stack<>();

    /**
     * Stack to store matrix column masks.
     *
     */
    private Stack<HashMap<Integer, Boolean>> colMaskStack = new Stack<>();

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
     */
    public SMatrix(int rows, int cols, Init initialization, int inputs, int outputs) {
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
     */
    public SMatrix(int rows, int cols, Init initialization) {
        this.rows = rows;
        this.cols = cols;
        init(initialization);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param data clones matrix data from given matrix data.
     */
    public SMatrix(HashMap<Integer, Double> data) {
        for (Integer index : data.keySet()) matrix.put(index, data.get(index));
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param data matrix data.
     * @param referTo if true creates matrix with reference to given matrix data otherwise clones the data.
     */
    public SMatrix(HashMap<Integer, Double> data, boolean referTo) {
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
     * Gets size (rows * columns) of matrix
     *
     * @return size of matrix.
     */
    public int getSize() {
        return rows * cols;
    }

    /**
     * Gets number of rows in matrix.
     *
     * @return number of rows in matrix.
     */
    public int getRows() {
        return !t ? rows : cols;
    }

    /**
     * Gets number of columns in matrix.
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

    /**
     * Matrix internal function used to set matrix masking of specific row and column.
     *
     * @param row row of value to be set.
     * @param col column of value to be set.
     * @param value defines if specific matrix row and column is masked (true) or not (false).
     */
    protected void setMaskValue(int row, int col, boolean value) {
        int curRow = !t ? row : col;
        int curCol = !t ? col : row;
        mask.put(curRow * cols + curCol, value);
    }

    /**
     * Matrix internal function used to get matrix masking of specific row and column.
     *
     * @param row row of value to be returned.
     * @param col column of value to be returned.
     * @return if specific matrix row and column if masked (true) or not (false).
     */
    protected boolean getMaskValue(int row, int col) {
        int curRow = !t ? row : col;
        int curCol = !t ? col : row;
        return mask.getOrDefault(curRow * cols + curCol, false);
    }

    /**
     * Clears and removes mask from this matrix.
     *
     */
    public void noMask() {
        hasMask = false;
        maskStack = new Stack<>();
        rowMaskStack = new Stack<>();
        colMaskStack = new Stack<>();
    }

    /**
     * Checks if mask if set.
     *
     * @throws MatrixException throws exception if mask is not set.
     */
    protected void checkMask() throws MatrixException {
        if (!hasMask) throw new MatrixException("Mask is not set");
    }

    /**
     * Resets current mask.
     *
     */
    public void resetMask() {
        hasMask = true;
        mask = new HashMap<>();
        rowMask = new HashMap<>();
        colMask = new HashMap<>();
    }

    /**
     * Pushes current mask into stack and optionally creates new mask for this matrix.<br>
     * Useful in operations where sequence of operations and taken between this matrix and other matrices.<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     * @throws MatrixException throws exception if mask is not set.
     */
    public void stackMask(boolean reset) throws MatrixException {
        checkMask();
        maskStack.push(mask);
        if (reset) mask = new HashMap<>();
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
     * Checks if row mask if set.
     *
     * @throws MatrixException throws exception if row mask is not set.
     */
    protected void checkRowMask() throws MatrixException {
        if (rowMask == null) throw new MatrixException("Row mask is not set");
    }

    /**
     * Pushes current row mask into stack and optionally creates new mask for this matrix.<br>
     * Useful in operations where sequence of operations and taken between this matrix and other matrices.<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     * @throws MatrixException throws exception if mask is not set.
     */
    public void stackRowMask(boolean reset) throws MatrixException {
        checkRowMask();
        rowMaskStack.push(rowMask);
        if (reset) rowMask = new HashMap<>();
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
        if (!t) rowMask.put(row, value);
        else colMask.put(row, value);
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
        return !t ? rowMask.getOrDefault(row, false) : colMask.getOrDefault(row, false);
    }

    /**
     * Checks if column mask if set.
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
        if (!t) colMask.put(col, value);
        else rowMask.put(col, value);
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
        return !t ? colMask.getOrDefault(col, false) : rowMask.getOrDefault(col, false);
    }

    /**
     * Pushes current column mask into stack and optionally creates new mask for this matrix.<br>
     * Useful in operations where sequence of operations and taken between this matrix and other matrices.<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     * @throws MatrixException throws exception if mask is not set.
     */
    public void stackColMask(boolean reset) throws MatrixException {
        checkColMask();
        colMaskStack.push(colMask);
        if (reset) colMask = new HashMap<>();
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
