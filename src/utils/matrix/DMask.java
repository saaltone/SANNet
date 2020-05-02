/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.matrix;

import java.util.Stack;

/**
 * Implements dense mask to mask dense matrices.
 *
 */
public class DMask extends Mask {

    /**
     * Defines number of rows in mask.
     *
     */
    private final int rows;

    /**
     * Defines number of columns in mask.
     *
     */
    private final int cols;

    /**
     * Mask data structure as two dimensional row column array-
     *
     */
    private boolean[][] mask;

    /**
     * Row mask data structure.
     *
     */
    private boolean[] rowMask;

    /**
     * Column mask data structure.
     *
     */
    private boolean[] colMask;

    /**
     * Stack to store masks.
     */
    private Stack<boolean[][]> maskStack = new Stack<>();

    /**
     * Stack to store row masks.
     *
     */
    private Stack<boolean[]> rowMaskStack = new Stack<>();

    /**
     * Stack to store column masks.
     *
     */
    private Stack<boolean[]> colMaskStack = new Stack<>();

    /**
     * Constructor for dense mask.
     *
     * @param rows defines number of rows in mask.
     * @param cols defines number of columns in mask.
     */
    public DMask(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        mask = new boolean[rows][cols];
        rowMask = new boolean[rows];
        colMask = new boolean[cols];
    }

    /**
     * Constructor for dense mask.
     *
     * @param data clones mask data from given mask data.
     */
    public DMask(boolean[][] data) {
        rows = data.length;
        cols = data[0].length;
        mask = data.clone();
        rowMask = new boolean[rows];
        colMask = new boolean[cols];
    }

    /**
     * Constructor for dense mask.
     *
     * @param data clones mask data from given mask data.
     * @param rowData clones row mask data from given row mask data.
     * @param colData clones column mask data from given column mask data.
     */
    public DMask(boolean[][] data, boolean[] rowData, boolean[] colData) {
        rows = data.length;
        cols = data[0].length;
        mask = data.clone();
        rowMask = rowData.clone();
        colMask = colData.clone();
    }

    /**
     * Constructor for dense mask.
     *
     * @param data clones mask data from given mask data.
     * @param rowData clones row mask data from given row mask data.
     * @param colData clones column mask data from given column mask data.
     * @param t if true mask if transposed otherwise false.
     * @param proba probability of masking.
     */
    public DMask(boolean[][] data, boolean[] rowData, boolean[] colData, boolean t, double proba) {
        rows = data.length;
        cols = data[0].length;
        mask = data.clone();
        rowMask = rowData.clone();
        colMask = colData.clone();
        this.t = t;
        this.proba = proba;
    }

    /**
     * Constructor for dense mask.
     *
     * @param data mask data.
     * @param referTo if true creates mask with reference to given mask data otherwise clones the data.
     */
    public DMask(boolean[][] data, boolean referTo) {
        this.rows = data.length;
        this.cols = data[0].length;
        if (referTo) mask = data;
        else mask = data.clone();
        rowMask = new boolean[rows];
        colMask = new boolean[cols];
    }

    /**
     * Retrieves copy of mask.
     *
     * @return copy of mask.
     */
    public Mask getCopy() {
        return new DMask(mask, rowMask, colMask, t, proba);
    }

    /**
     * Returns size (rows * columns) of mask
     *
     * @return size of mask.
     */
    public int getSize() {
        return rows * cols;
    }

    /**
     * Returns number of rows in mask.
     *
     * @return number of rows in mask.
     */
    public int getRows() {
        return !t ? rows : cols;
    }

    /**
     * Returns number of columns in mask.
     *
     * @return number of columns in mask.
     */
    public int getCols() {
        return !t ? cols : rows;
    }

    /**
     * Sets masking of specific row and column.
     *
     * @param row row of value to be get.
     * @param col column of value to be get.
     * @param value defines if specific row and column is masked (true) or not (false).
     */
    public void setMask(int row, int col, boolean value) {
        mask[!t ? row : col][!t ? col : row] = value;
    }

    /**
     * Returns masking of specific row and column.
     *
     * @param row row of value to be returned.
     * @param col column of value to be returned.
     * @return if specific row and column is masked (true) or not (false).
     */
    public boolean getMask(int row, int col) {
        return mask[!t ? row : col][!t ? col : row];
    }

    /**
     * Clears and removes mask.
     *
     */
    public void clear() {
        mask = new boolean[rows][cols];
        rowMask = new boolean[rows];
        colMask = new boolean[cols];
        maskStack = new Stack<>();
        rowMaskStack = new Stack<>();
        colMaskStack = new Stack<>();
    }

    /**
     * Pushes current mask into stack and optionally creates new mask.<br>
     * Useful in operations where sequence of operations are taken between matrices.<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     */
    public void stackMask(boolean reset) {
        maskStack.push(mask);
        if (reset) mask = new boolean[mask.length][mask[0].length];
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
     * Pushes current row mask into stack and optionally creates new mask.<br>
     * Useful in operations where sequence of operations are taken between matrices.<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     */
    public void stackRowMask(boolean reset) {
        rowMaskStack.push(rowMask);
        if (reset) rowMask = new boolean[mask.length];
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
     */
    public void setRowMask(int row, boolean value) {
        if (!t) rowMask[row] = value;
        else colMask[row] = value;
    }

    /**
     * Returns mask value for row mask.
     *
     * @param row row of mask to be returned.
     * @return true if row mask is set otherwise false.
     */
    public boolean getRowMask(int row) {
        return !t ? rowMask[row] : colMask[row];
    }

    /**
     * Sets mask value for column mask.
     *
     * @param col column of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setColMask(int col, boolean value) {
        if (!t) colMask[col] = value;
        else rowMask[col] = value;
    }

    /**
     * Returns mask value for column mask.
     *
     * @param col column of mask to be returned.
     * @return true if row mask is set otherwise false.
     */
    public boolean getColMask(int col) {
        return !t ? colMask[col] : rowMask[col];
    }

    /**
     * Pushes current column mask into stack and optionally creates new mask.<br>
     * Useful in operations where sequence of operations are taken between matrices.<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     */
    public void stackColMask(boolean reset) {
        colMaskStack.push(colMask);
        if (reset) colMask = new boolean[mask[0].length];
    }

    /**
     * Pops column mask from mask stack.
     *
     * @throws MatrixException throws exception if mask is not set or column mask stack is empty.
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
