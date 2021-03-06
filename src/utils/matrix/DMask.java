/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.matrix;

/**
 * Implements dense mask to mask dense matrices.<br>
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
     * Constructor for dense mask.
     *
     * @param rows defines number of rows in mask.
     * @param columns defines number of columns in mask.
     */
    public DMask(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        mask = new boolean[rows][columns];
    }

    /**
     * Constructor for dense mask.
     *
     * @param data clones mask data from given mask data.
     * @param probability probability of masking.
     */
    protected DMask(boolean[][] data, double probability) {
        rows = data.length;
        columns = data[0].length;
        mask = data.clone();
        this.probability = probability;
    }

    /**
     * Returns new mask of same dimensions.
     *
     * @return new mask of same dimensions.
     */
    public Mask getNewMask() {
        return new DMask(getRows(), getColumns());
    }

    /**
     * Returns new mask of same dimensions optionally as transposed.
     *
     * @param asTransposed if true returns new mask as transposed otherwise with unchanged dimensions.
     * @return new mask of same dimensions.
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
        return new DMask(mask, probability);
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
     * Sets mask at specific row and column.
     *
     * @param row row of value to be get.
     * @param column column of value to be get.
     * @param value defines if specific row and column is masked (true) or not (false).
     */
    public void setMask(int row, int column, boolean value) {
        mask[row][column] = value;
    }

    /**
     * Returns mask at specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @return if specific row and column is masked (true) or not (false).
     */
    public boolean getMask(int row, int column) {
        return mask[row][column];
    }

    /**
     * Clears mask.
     *
     */
    public void clear() {
        mask = new boolean[rows][columns];
    }

    /**
     * Sets mask value for row mask.
     *
     * @param row row of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setRowMask(int row, boolean value) {
        for (int column = 0; column < mask[0].length; column++) mask[row][column] = value;
    }

    /**
     * Sets mask value for column mask.
     *
     * @param column column of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setColumnMask(int column, boolean value) {
        for (int row = 0; row < mask.length; row++) mask[row][column] = value;
    }

}
