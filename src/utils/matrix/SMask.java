/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.matrix;

import java.util.HashMap;

/**
 * Implements sparse mask to mask sparse matrices.
 *
 */
public class SMask extends AbstractMask {

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
     * @param probability probability of masking.
     */
    protected SMask(int rows, int columns, HashMap<Integer, Boolean> data, HashMap<Integer, Boolean> rowData, HashMap<Integer, Boolean> columnData, double probability) {
        this.rows = rows;
        this.columns = columns;
        mask.putAll(data);
        rowMask.putAll(rowData);
        columnMask.putAll(columnData);
        this.probability = probability;
    }

    /**
     * Returns new mask of dimensions rows x columns.<br>
     *
     * @return new mask of dimensions rows x columns.
     */
    public Mask getNewMask() {
        return new SMask(getRows(), getColumns());
    }

    /**
     * Returns new mask of dimensions rows x columns.<br>
     *
     * @param asTransposed if true returns new mask as transposed otherwise with unchanged dimensions.
     * @return new mask of dimensions rows x columns.
     */
    public Mask getNewMask(boolean asTransposed) {
        return !asTransposed ? new SMask(getRows(), getColumns()) : new SMask(getColumns(), getRows());
    }

    /**
     * Retrieves copy of mask.
     *
     * @return copy of mask.
     */
    public Mask getCopy() {
        return new SMask(rows, columns, mask, rowMask, columnMask, probability);
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
        return rows;
    }

    /**
     * Returns number of columns in mask.
     *
     * @return number of columns in mask.
     */
    public int getColumns() {
        return columns;
    }

    /**
     * Internal function used to set masking of specific row and column.
     *
     * @param row row of value to be set.
     * @param column column of value to be set.
     * @param value defines if specific row and column is masked (true) or not (false).
     */
    public void setMask(int row, int column, boolean value) {
        mask.put(row * columns + column, value);
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
        return mask.getOrDefault(row * columns + column, false);
    }

    /**
     * Clears and removes mask.
     *
     */
    public void clear() {
        mask = new HashMap<>();
        rowMask = new HashMap<>();
        columnMask = new HashMap<>();
    }

    /**
     * Sets mask value for row mask.
     *
     * @param row row of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setRowMask(int row, boolean value) {
        rowMask.put(row, value);
    }

    /**
     * Returns mask value for row mask.
     *
     * @param row row of mask to be returned.
     * @return true if row mask is set otherwise false.
     */
    public boolean getRowMask(int row) {
        if (mask == null) return false;
        return rowMask.getOrDefault(row, false);
    }

    /**
     * Sets mask value for column mask.
     *
     * @param column column of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setColumnMask(int column, boolean value) {
        columnMask.put(column, value);
    }

    /**
     * Returns mask value for column mask.
     *
     * @param column column of mask to be returned.
     * @return true if row mask is set otherwise false.
     */
    public boolean getColumnMask(int column) {
        if (mask == null) return false;
        return columnMask.getOrDefault(column, false);
    }

}
