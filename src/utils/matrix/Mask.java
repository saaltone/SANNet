/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.matrix;

/**
 * Interface defining masking of matrices.<br>
 *
 */
public interface Mask {

    /**
     * Returns new mask of same dimensions.
     *
     * @return new mask of same dimensions.
     */
    Mask getNewMask();

    /**
     * Returns new mask of same dimensions optionally as transposed.
     *
     * @param asTransposed if true returns new mask as transposed otherwise with unchanged dimensions.
     * @return new mask of same dimensions.
     */
    Mask getNewMask(boolean asTransposed);

    /**
     * Creates new mask with object reference to the mask data of this mask.
     *
     * @return newly created reference mask.
     * @throws MatrixException throws exception if cloning of mask fails.
     */
    Mask reference() throws MatrixException;

    /**
     * Creates new mask with object full copy of this mask.
     *
     * @return newly created reference mask.
     */
    Mask copy();

    /**
     * Retrieves copy of mask.
     *
     * @return copy of mask.
     */
    Mask getCopy();

    /**
     * Transposes mask.
     *
     * @return reference to this mask but with transposed that is flipped rows and columns.
     */
    Mask transpose();

    /**
     * Returns size (rows * columns) of mask.<br>
     *
     * @return size of mask.
     */
    int size();

    /**
     * Returns number of rows in mask.<br>
     *
     * @return number of rows in mask.
     */
    int getRows();

    /**
     * Returns number of columns in mask.<br>
     *
     * @return number of columns in mask.
     */
    int getColumns();

    /**
     * Clears mask.<br>
     *
     */
    void clear();

    /**
     * Checks if mask is set at specific row and column
     *
     * @param row row to be checked.
     * @param column column to be checked.
     * @return result of mask check.
     */
    boolean isMasked(int row, int column);

    /**
     * Sets bernoulli probability to mask specific element.
     *
     * @param probability masking probability between 0 (0%) and 1 (100%).
     * @throws MatrixException throws exception if masking probability is not between 0 and 1.
     */
    void setProbability(double probability) throws MatrixException;

    /**
     * Returns current bernoulli masking probability.
     *
     * @return masking probability.
     */
    double getProbability();

    /**
     * Sets mask at specific row and column.
     *
     * @param row row of mask to be set.
     * @param column column of mask to be set.
     * @param value sets mask if true otherwise unsets mask.
     */
    void setMask(int row, int column, boolean value);

    /**
     * Returns mask at specific row and column.
     *
     * @param row row of mask to be returned.
     * @param column column of mask to be returned.
     * @return true if mask is set otherwise false.
     */
    boolean getMask(int row, int column);

    /**
     * Sets masking for this mask with given bernoulli probability.
     *
     */
    void maskByProbability();

    /**
     * Sets mask value for row mask.
     *
     * @param row row of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    void setRowMask(int row, boolean value);

    /**
     * Sets row masking for this mask with given bernoulli probability.
     *
     */
    void maskRowByProbability();

    /**
     * Sets mask value for column mask.
     *
     * @param column column of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    void setColumnMask(int column, boolean value);

    /**
     * Sets column masking for this mask with given bernoulli probability.
     *
     */
    void maskColumnByProbability();

    /**
     * Prints mask in row and column format.
     *
     */
    void print();

    /**
     * Prints size (rows x columns) of mask.
     *
     */
    void printSize();

}
