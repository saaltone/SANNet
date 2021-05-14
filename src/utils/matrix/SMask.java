/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.matrix;

import java.util.HashMap;

/**
 * Implements sparse mask to mask sparse matrices.<br>
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
     * @param probability probability of masking.
     */
    protected SMask(int rows, int columns, HashMap<Integer, Boolean> data, double probability) {
        this.rows = rows;
        this.columns = columns;
        mask.putAll(data);
        this.probability = probability;
    }

    /**
     * Returns new mask of same dimensions.
     *
     * @return new mask of same dimensions.
     */
    public Mask getNewMask() {
        return new SMask(getRows(), getColumns());
    }

    /**
     * Returns new mask of same dimensions optionally as transposed.
     *
     * @param asTransposed if true returns new mask as transposed otherwise with unchanged dimensions.
     * @return new mask of same dimensions.
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
        return new SMask(rows, columns, mask, probability);
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
     * @param row row of value to be set.
     * @param column column of value to be set.
     * @param value defines if specific row and column is masked (true) or not (false).
     */
    public void setMask(int row, int column, boolean value) {
        mask.put(row * columns + column, value);
    }

    /**
     * Returns mask at specific row and column.
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
    }

    /**
     * Sets mask value for row mask.
     *
     * @param row row of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setRowMask(int row, boolean value) {
        for (int column = 0; column < columns; column++) mask.put(row * columns + column, value);
    }

    /**
     * Sets mask value for column mask.
     *
     * @param column column of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setColumnMask(int column, boolean value) {
        for (int row = 0; row < rows; row++) mask.put(row * columns + column, value);
    }

}
