/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix;

/**
 * Implements dense mask to mask dense matrices.<br>
 *
 */
public class DMask extends AbstractMask {

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
        super(rows, columns);
        mask = new boolean[rows][columns];
    }

    /**
     * Constructor for dense mask.
     *
     * @param data clones mask data from given mask data.
     * @param probability probability of masking.
     */
    protected DMask(boolean[][] data, double probability) {
        super(data.length, data[0].length, probability);
        mask = data.clone();
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
        return new DMask(mask, getProbability());
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
     * Resets mask.
     *
     */
    public void reset() {
        mask = new boolean[getRows()][getColumns()];
    }

}
