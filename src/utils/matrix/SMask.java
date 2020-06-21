/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.matrix;

import java.util.HashMap;
import java.util.Stack;

/**
 * Implements sparse mask to mask sparse matrices.
 *
 */
public class SMask extends Mask {

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
    private HashMap<Integer, Boolean> columnMask = new HashMap<>();

    /**
     * Stack to store masks.
     */
    private Stack<HashMap<Integer, Boolean>> maskStack = new Stack<>();

    /**
     * Stack to store row masks.
     *
     */
    private Stack<HashMap<Integer, Boolean>> rowMaskStack = new Stack<>();

    /**
     * Stack to store column masks.
     *
     */
    private Stack<HashMap<Integer, Boolean>> columnMaskStack = new Stack<>();

    /**
     * Constructor for sparse mask.
     *
     * @param rows defines number of rows in mask.
     * @param columns defines number of columns in mask.
     */
    public SMask(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
    }

    /**
     * Constructor for sparse mask.
     *
     * @param rows defines number of rows in mask.
     * @param columns defines number of columns in mask.
     * @param data mask data.
     * @param rowData row mask data.
     * @param columnData column mask data.
     * @param transposed if true mask if transposed otherwise false.
     * @param probability probability of masking.
     */
    protected SMask(int rows, int columns, HashMap<Integer, Boolean> data, HashMap<Integer, Boolean> rowData, HashMap<Integer, Boolean> columnData, boolean transposed, double probability) {
        this.rows = rows;
        this.columns = columns;
        mask.putAll(data);
        rowMask.putAll(rowData);
        columnMask.putAll(columnData);
        this.isTransposed = transposed;
        this.probability = probability;
    }

    /**
     * Retrieves copy of mask.
     *
     * @return copy of mask.
     */
    public Mask getCopy() {
        return new SMask(rows, columns, mask, rowMask, columnMask, isTransposed, probability);
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
     * Internal function used to set masking of specific row and column.
     *
     * @param row row of value to be set.
     * @param column column of value to be set.
     * @param value defines if specific row and column is masked (true) or not (false).
     */
    public void setMask(int row, int column, boolean value) {
        int currentRow = !isTransposed ? row : column;
        int currentColumn = !isTransposed ? column : row;
        mask.put(currentRow * columns + currentColumn, value);
    }

    /**
     * Internal function used to get masking of specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @return if specific row and column if masked (true) or not (false).
     */
    public boolean getMask(int row, int column) {
        if (mask == null) return false;
        int currentRow = !isTransposed ? row : column;
        int currentColumn = !isTransposed ? column : row;
        return mask.getOrDefault(currentRow * columns + currentColumn, false);
    }

    /**
     * Clears and removes mask.
     *
     */
    public void clear() {
        mask = new HashMap<>();
        rowMask = new HashMap<>();
        columnMask = new HashMap<>();
        maskStack = new Stack<>();
        rowMaskStack = new Stack<>();
        columnMaskStack = new Stack<>();
    }

    /**
     * Pushes current mask into stack and optionally creates new mask.<br>
     * Useful in operations where sequence of operations and taken between matrices.<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     */
    public void stackMask(boolean reset) {
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
     * Pushes current row mask into stack and optionally creates new mask.<br>
     * Useful in operations where sequence of operations and taken between matrices.<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     */
    public void stackRowMask(boolean reset) {
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
     */
    public void setRowMask(int row, boolean value) {
        if (!isTransposed) rowMask.put(row, value);
        else columnMask.put(row, value);
    }

    /**
     * Returns mask value for row mask.
     *
     * @param row row of mask to be returned.
     * @return true if row mask is set otherwise false.
     */
    public boolean getRowMask(int row) {
        if (mask == null) return false;
        return !isTransposed ? rowMask.getOrDefault(row, false) : columnMask.getOrDefault(row, false);
    }

    /**
     * Sets mask value for column mask.
     *
     * @param column column of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setColumnMask(int column, boolean value) {
        if (!isTransposed) columnMask.put(column, value);
        else rowMask.put(column, value);
    }

    /**
     * Returns mask value for column mask.
     *
     * @param column column of mask to be returned.
     * @return true if row mask is set otherwise false.
     */
    public boolean getColumnMask(int column) {
        if (mask == null) return false;
        return !isTransposed ? columnMask.getOrDefault(column, false) : rowMask.getOrDefault(column, false);
    }

    /**
     * Pushes current column mask into stack and optionally creates new mask.<br>
     * Useful in operations where sequence of operations and taken between matrices.<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     */
    public void stackColumnMask(boolean reset) {
        columnMaskStack.push(columnMask);
        if (reset) columnMask = new HashMap<>();
    }

    /**
     * Pops column mask from mask stack.
     *
     * @throws MatrixException throws exception if column mask stack is empty.
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
