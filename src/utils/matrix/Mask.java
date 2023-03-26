/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
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
     * @throws MatrixException throws exception if masking probability is not between 0 and 1.
     */
    Mask copy() throws MatrixException;

    /**
     * Transposes mask.
     *
     * @return reference to this mask but with transposed that is flipped rows and columns.
     * @throws MatrixException throws exception if cloning of mask fails.
     */
    Mask transpose() throws MatrixException;

    /**
     * Checks if mask is transposed.
     *
     * @return true is mask is transposed otherwise false.
     */
    boolean isTransposed();

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
     * Returns depth of mask.<br>
     *
     * @return depth of mask.
     */
    int getDepth();

    /**
     * Resets mask.
     *
     */
    void reset();

    /**
     * Checks if mask is set at specific row, column and depth
     *
     * @param row row to be checked.
     * @param column column to be checked.
     * @param depth depth to be checked.
     * @return result of mask check.
     */
    boolean isMasked(int row, int column, int depth);

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
     * @param depth depth of mask to be set.
     * @param value sets mask if true otherwise unsets mask.
     */
    void setMask(int row, int column, int depth, boolean value);

    /**
     * Returns mask at specific row and column.
     *
     * @param row row of mask to be returned.
     * @param column column of mask to be returned.
     * @param depth depth of mask to be returned.
     * @return true if mask is set otherwise false.
     */
    boolean getMask(int row, int column, int depth);

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
     * @param value if true sets column mask otherwise unsets mask.
     */
    void setColumnMask(int column, boolean value);

    /**
     * Sets column masking for this mask with given bernoulli probability.
     *
     */
    void maskColumnByProbability();

    /**
     * Sets mask value for depth mask.
     *
     * @param depth depth of mask to be set.
     * @param value if true sets depth mask otherwise unsets mask.
     */
    void setDepthMask(int depth, boolean value);

    /**
     * Sets depth masking for this mask with given bernoulli probability.
     *
     */
    void maskDepthByProbability();

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
