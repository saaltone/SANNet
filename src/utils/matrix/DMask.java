/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.matrix;

/**
 * Implements dense mask for dense matrices.<br>
 *
 */
public class DMask extends AbstractMask {

    /**
     * Defines mask data structure using 2-dimensional row column array.
     *
     */
    private boolean[] mask;

    /**
     * Constructor for dense mask.
     *
     * @param rows defines number of rows in mask.
     * @param columns defines number of columns in mask.
     * @param depth defines depth of mask.
     */
    public DMask(int rows, int columns, int depth) {
        this(rows, columns, depth, false);
    }

    /**
     * Constructor for dense mask.
     *
     * @param rows defines number of rows in mask.
     * @param columns defines number of columns in mask.
     * @param depth defines depth of mask.
     * @param isTransposed is true mask is transposed otherwise false.
     */
    public DMask(int rows, int columns, int depth, boolean isTransposed) {
        super(rows, columns, depth, isTransposed);
        mask = new boolean[rows * columns * depth];
    }

    /**
     * Constructor for dense mask.
     *
     * @param rows defines number of rows in mask.
     * @param columns defines number of columns in mask.
     * @param depth defines depth of mask.
     * @param data clones mask data from given mask data.
     * @param probability probability of masking.
     * @param isTransposed is true mask is transposed otherwise false.
     * @throws MatrixException throws exception if masking probability is not between 0 and 1.
     */
    private DMask(int rows, int columns, int depth, boolean[] data, double probability, boolean isTransposed) throws MatrixException {
        super(rows, columns, depth, isTransposed, probability);
        mask = data.clone();
    }

    /**
     * Returns new mask of same dimensions.
     *
     * @return new mask of same dimensions.
     */
    public Mask getNewMask() {
        return new DMask(getRows(), getColumns(), getDepth());
    }

    /**
     * Returns copy of mask.
     *
     * @return copy of mask.
     * @throws MatrixException throws exception if masking probability is not between 0 and 1.
     */
    public Mask applyCopy() throws MatrixException {
        return new DMask(getPureRows(), getPureColumns(), getPureDepth(), mask, getProbability(), isTransposed());
    }

    /**
     * Applies transpose operation.
     *
     * @return transposed mask.
     */
    protected Mask applyTranspose() throws MatrixException {
        return new DMask(getPureRows(), getPureColumns(), getPureDepth(), mask, getProbability(), !isTransposed());
    }

    /**
     * Sets mask at specific row and column.
     *
     * @param row row of value to be set.
     * @param column column of value to be set.
     * @param depth depth of value to be set.
     * @param value defines if specific row and column is masked (true) or not (false).
     */
    public void setMask(int row, int column, int depth, boolean value) {
        mask[getArrayIndex(row, column, depth)] = value;
    }

    /**
     * Returns mask at specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @param depth depth of value to be returned.
     * @return if specific row and column is masked (true) or not (false).
     */
    public boolean getMask(int row, int column, int depth) {
        return mask[getArrayIndex(row, column, depth)];
    }

    /**
     * Resets mask.
     *
     */
    public void reset() {
        mask = new boolean[getRows() * getColumns() * getDepth()];
    }

}
