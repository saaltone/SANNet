/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.matrix;

/**
 * Implements dense mask to mask dense matrices.
 *
 */
public class DMask extends AbstractMask {

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
     * @param probability probability of masking.
     */
    protected DMask(boolean[][] data, boolean[] rowData, boolean[] columnData, double probability) {
        rows = data.length;
        columns = data[0].length;
        mask = data.clone();
        rowMask = rowData.clone();
        columnMask = columnData.clone();
        this.probability = probability;
    }

    /**
     * Returns new mask of dimensions rows x columns.<br>
     *
     * @return new mask of dimensions rows x columns.
     */
    public Mask getNewMask() {
        return new DMask(getRows(), getColumns());
    }

    /**
     * Returns new mask of dimensions rows x columns.<br>
     *
     * @param asTransposed if true returns new mask as transposed otherwise with unchanged dimensions.
     * @return new mask of dimensions rows x columns.
     */
    public Mask getNewMask(boolean asTransposed) {
        return !asTransposed ? new DMask(getRows(), getColumns()) : new DMask(getColumns(), getRows());
    }

    /**
     * Returns copy of mask.
     *
     * @return copy of mask.
     */
    public Mask getCopy() {
        return new DMask(mask, rowMask, columnMask, probability);
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
     * Sets masking of specific row and column.
     *
     * @param row row of value to be get.
     * @param column column of value to be get.
     * @param value defines if specific row and column is masked (true) or not (false).
     */
    public void setMask(int row, int column, boolean value) {
        mask[row][column] = value;
    }

    /**
     * Returns masking of specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @return if specific row and column is masked (true) or not (false).
     */
    public boolean getMask(int row, int column) {
        return mask[row][column];
    }

    /**
     * Clears and removes mask.
     *
     */
    public void clear() {
        mask = new boolean[rows][columns];
        rowMask = new boolean[rows];
        columnMask = new boolean[columns];
    }

    /**
     * Sets mask value for row mask.
     *
     * @param row row of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setRowMask(int row, boolean value) {
        rowMask[row] = value;
    }

    /**
     * Returns mask value for row mask.
     *
     * @param row row of mask to be returned.
     * @return true if row mask is set otherwise false.
     */
    public boolean getRowMask(int row) {
        return rowMask[row];
    }

    /**
     * Sets mask value for column mask.
     *
     * @param column column of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setColumnMask(int column, boolean value) {
        columnMask[column] = value;
    }

    /**
     * Returns mask value for column mask.
     *
     * @param column column of mask to be returned.
     * @return true if row mask is set otherwise false.
     */
    public boolean getColumnMask(int column) {
        return columnMask[column];
    }

}
