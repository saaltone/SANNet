/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix;

import java.util.HashMap;

/**
 * Implements sparse mask for sparse matrices.<br>
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
     * @param depth defines depth of mask.
     */
    public SMask(int rows, int columns, int depth) {
        this(rows, columns, depth, false);
    }

    /**
     * Constructor for sparse mask.
     *
     * @param rows defines number of rows in mask.
     * @param columns defines number of columns in mask.
     * @param depth defines depth of mask.
     * @param isTransposed is true mask is transposed otherwise false.
     */
    public SMask(int rows, int columns, int depth, boolean isTransposed) {
        super(rows, columns, depth, isTransposed);
    }

    /**
     * Constructor for sparse mask.
     *
     * @param rows defines number of rows in mask.
     * @param columns defines number of columns in mask.
     * @param data mask data.
     * @param probability probability of masking.
     * @param isTransposed is true mask is transposed otherwise false.
     * @throws MatrixException throws exception if masking probability is not between 0 and 1.
     */
    private SMask(int rows, int columns, int depth, HashMap<Integer, Boolean> data, double probability, boolean isTransposed) throws MatrixException {
        super(rows, columns, depth, isTransposed, probability);
        mask.putAll(data);
    }

    /**
     * Returns new mask of same dimensions.
     *
     * @return new mask of same dimensions.
     */
    public Mask getNewMask() {
        return new SMask(getPureRows(), getPureColumns(), getPureDepth(), isTransposed());
    }

    /**
     * Retrieves copy of mask.
     *
     * @return copy of mask.
     * @throws MatrixException throws exception if masking probability is not between 0 and 1.
     */
    public Mask applyCopy() throws MatrixException {
        return new SMask(getPureRows(), getPureColumns(), getPureDepth(), mask, getProbability(), isTransposed());
    }

    /**
     * Applies transpose operation.
     *
     * @return transposed mask.
     * @throws MatrixException throws exception if masking probability is not between 0 and 1.
     */
    protected Mask applyTranspose() throws MatrixException {
        return new SMask(getPureRows(), getPureColumns(), getPureDepth(), mask, getProbability(), !isTransposed());
    }

    /**
     * Sets mask at specific row and column.
     *
     * @param row row of value to be set.
     * @param column column of value to be set.
     * @param value defines if specific row and column is masked (true) or not (false).
     */
    public void setMask(int row, int column, int depth, boolean value) {
        mask.put(getArrayIndex(row, column, depth), value);
    }

    /**
     * Returns mask at specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @param depth depth of value to be returned.
     * @return if specific row and column if masked (true) or not (false).
     */
    public boolean getMask(int row, int column, int depth) {
        if (mask == null) return false;
        return mask.getOrDefault(getArrayIndex(row, column, depth), false);
    }

    /**
     * Resets mask.
     *
     */
    public void reset() {
        mask = new HashMap<>();
    }

}
