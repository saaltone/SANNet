/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix;

import java.util.HashMap;

/**
 * Implements sparse mask to mask sparse matrices.<br>
 *
 */
public class SMask extends AbstractMask {

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
        super(rows, columns);
    }

    /**
     * Constructor for sparse mask.
     *
     * @param rows defines number of rows in mask.
     * @param columns defines number of columns in mask.
     * @param data mask data.
     * @param probability probability of masking.
     * @param isTransposed is true mask is transposed otherwise false.
     */
    private SMask(int rows, int columns, HashMap<Integer, Boolean> data, double probability, boolean isTransposed) {
        super(rows, columns, probability, isTransposed);
        mask.putAll(data);
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
        return new SMask(!isTransposed ? getRows() : getColumns(), !isTransposed ? getColumns() : getRows(), mask, getProbability(), isTransposed);
    }

    /**
     * Sets mask at specific row and column.
     *
     * @param row row of value to be set.
     * @param column column of value to be set.
     * @param value defines if specific row and column is masked (true) or not (false).
     */
    public void setMask(int row, int column, boolean value) {
        mask.put(row * (!isTransposed ? getColumns() : getRows()) + column, value);
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
        return mask.getOrDefault(row * (!isTransposed ? getColumns() : getRows()) + column, false);
    }

    /**
     * Resets mask.
     *
     */
    public void reset() {
        mask = new HashMap<>();
    }

}
