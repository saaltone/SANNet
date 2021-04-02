/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

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
    private final int columns;

    /**
     * Mask data structure as two dimensional row column array.
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
    private boolean[] columnMask;

    /**
     * Stack to store masks.
     *
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
    private Stack<boolean[]> columnMaskStack = new Stack<>();

    /**
     * Constructor for dense mask.
     *
     * @param rows defines number of rows in mask.
     * @param columns defines number of columns in mask.
     */
    public DMask(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        mask = new boolean[rows][columns];
        rowMask = new boolean[rows];
        columnMask = new boolean[columns];
    }

    /**
     * Constructor for dense mask.
     *
     * @param data clones mask data from given mask data.
     * @param rowData clones row mask data from given row mask data.
     * @param columnData clones column mask data from given column mask data.
     * @param isTransposed if true mask if transposed otherwise false.
     * @param probability probability of masking.
     */
    protected DMask(boolean[][] data, boolean[] rowData, boolean[] columnData, boolean isTransposed, double probability) {
        rows = data.length;
        columns = data[0].length;
        mask = data.clone();
        rowMask = rowData.clone();
        columnMask = columnData.clone();
        this.isTransposed = isTransposed;
        this.probability = probability;
    }

    /**
     * Returns copy of mask.
     *
     * @return copy of mask.
     */
    public Mask getCopy() {
        return new DMask(mask, rowMask, columnMask, isTransposed, probability);
    }

    /**
     * Returns size (rows * columns) of mask
     *
     * @return size of mask.
     */
    public int size() {
        return rows * columns;
    }

    /**
     * Returns number of rows in mask.
     *
     * @return number of rows in mask.
     */
    public int getRows() {
        return !isTransposed ? rows : columns;
    }

    /**
     * Returns number of columns in mask.
     *
     * @return number of columns in mask.
     */
    public int getColumns() {
        return !isTransposed ? columns : rows;
    }

    /**
     * Sets masking of specific row and column.
     *
     * @param row row of value to be get.
     * @param column column of value to be get.
     * @param value defines if specific row and column is masked (true) or not (false).
     */
    public void setMask(int row, int column, boolean value) {
        mask[!isTransposed ? row : column][!isTransposed ? column : row] = value;
    }

    /**
     * Returns masking of specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @return if specific row and column is masked (true) or not (false).
     */
    public boolean getMask(int row, int column) {
        return mask[!isTransposed ? row : column][!isTransposed ? column : row];
    }

    /**
     * Clears and removes mask.
     *
     */
    public void clear() {
        mask = new boolean[rows][columns];
        rowMask = new boolean[rows];
        columnMask = new boolean[columns];
        maskStack = new Stack<>();
        rowMaskStack = new Stack<>();
        columnMaskStack = new Stack<>();
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
        if (!isTransposed) rowMask[row] = value;
        else columnMask[row] = value;
    }

    /**
     * Returns mask value for row mask.
     *
     * @param row row of mask to be returned.
     * @return true if row mask is set otherwise false.
     */
    public boolean getRowMask(int row) {
        return !isTransposed ? rowMask[row] : columnMask[row];
    }

    /**
     * Sets mask value for column mask.
     *
     * @param column column of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setColumnMask(int column, boolean value) {
        if (!isTransposed) columnMask[column] = value;
        else rowMask[column] = value;
    }

    /**
     * Returns mask value for column mask.
     *
     * @param column column of mask to be returned.
     * @return true if row mask is set otherwise false.
     */
    public boolean getColumnMask(int column) {
        return !isTransposed ? columnMask[column] : rowMask[column];
    }

    /**
     * Pushes current column mask into stack and optionally creates new mask.<br>
     * Useful in operations where sequence of operations are taken between matrices.<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     */
    public void stackColumnMask(boolean reset) {
        columnMaskStack.push(columnMask);
        if (reset) columnMask = new boolean[mask[0].length];
    }

    /**
     * Pops column mask from mask stack.
     *
     * @throws MatrixException throws exception if mask is not set or column mask stack is empty.
     */
    public void unstackColumnMask() throws MatrixException {
        if (columnMaskStack.isEmpty()) throw new MatrixException("Column mask stack is empty.");
        columnMask = columnMaskStack.pop();
    }

    /**
     * Returns size of a column mask stack.
     *
     * @return size of column mask stack.
     */
    public int columnMaskStackSize() {
        return columnMaskStack.size();
    }

    /**
     * Clears column mask stack.
     *
     */
    public void clearColumnMaskStack() {
        columnMaskStack = new Stack<>();
    }

}
