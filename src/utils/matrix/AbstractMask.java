/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix;

import java.io.Serial;
import java.io.Serializable;
import java.util.Random;

/**
 * Abstract class that implements common operations for masking.<br>
 *
 */
public abstract class AbstractMask implements Cloneable, Serializable, Mask {

    @Serial
    private static final long serialVersionUID = -4902569287054022460L;

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
     * Bernoulli probability for selecting if entry (row, column) is masked or not.
     *
     */
    private double probability = 0;

    /**
     * Random function for mask class.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for abstract mask.
     *
     * @param rows number of rows in mask.
     * @param columns number of columns in mask.
     */
    public AbstractMask(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
    }

    /**
     * Constructor for abstract mask.
     *
     * @param rows number of rows in mask.
     * @param columns number of columns in mask.
     * @param probability probability of masking.
     */
    public AbstractMask(int rows, int columns, double probability) {
        this.rows = rows;
        this.columns = columns;
        this.probability = probability;
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
     * Creates new mask with object reference to the mask data of this mask.
     *
     * @return newly created reference mask.
     * @throws MatrixException throws exception if cloning of mask fails.
     */
    public Mask reference() throws MatrixException {
        Mask newMask;
        // Make shallow copy of mask leaving references internal objects which are shared.
        try {
            newMask = (Mask)super.clone();
        } catch (CloneNotSupportedException exception) {
            throw new MatrixException("Cloning of mask failed.");
        }
        return newMask;
    }

    /**
     * Creates new mask with full copy of this mask.
     *
     * @return newly created mask copy.
     */
    public Mask copy() {
        return getCopy();
    }

    /**
     * Transposes mask.
     *
     * @return reference to this mask but as transposed with flipped rows and columns.
     */
    public Mask transpose() {
        Mask transposedMask = getNewMask(true);
        int rows = getRows();
        int columns = getColumns();
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                if (getMask(row, column)) transposedMask.setMask(column, row, getMask(row, column));
            }
        }
        return transposedMask;
    }

    /**
     * Checks if mask is set at specific row and column
     *
     * @param row row.
     * @param column column.
     * @return return true if mask is set at row and column.
     */
    public boolean isMasked(int row, int column) {
        return getMask(row, column);
    }

    /**
     * Sets bernoulli probability to mask specific row and column.
     *
     * @param probability masking probability between 0 (0%) and 1 (100%).
     * @throws MatrixException throws exception if masking probability is not between 0 and 1.
     */
    public void setProbability(double probability) throws MatrixException {
        if (probability < 0 || probability > 1) throw new MatrixException("Masking probability must be between 0 and 1.");
        this.probability = probability;
    }

    /**
     * Returns current bernoulli masking probability.
     *
     * @return masking probability.
     */
    public double getProbability() {
        return probability;
    }

    /**
     * Returns true with defined masking probability.
     *
     * @return true with defined masking probability.
     */
    private boolean isMaskedByProbability() {
        return random.nextDouble() > probability;
    }

    /**
     * Sets masking with given bernoulli probability for each row and column.
     *
     */
    public void maskByProbability() {
        int rows = getRows();
        int columns = getColumns();
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                setMask(row, column, isMaskedByProbability());
            }
        }
    }

    /**
     * Sets row masking for this mask with given bernoulli probability.
     *
     */
    public void maskRowByProbability() {
        int rows = getRows();
        for (int row = 0; row < rows; row++) {
            setRowMask(row, isMaskedByProbability());
        }
    }

    /**
     * Sets column masking for this mask with given bernoulli probability.
     *
     */
    public void maskColumnByProbability() {
        int columns = getColumns();
        for (int column = 0; column < columns; column++) {
            setColumnMask(column, isMaskedByProbability());
        }
    }

    /**
     * Sets mask value for row mask.
     *
     * @param row row of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setRowMask(int row, boolean value) {
        int columns = getColumns();
        for (int column = 0; column < columns; column++) setMask(row, column, value);
    }

    /**
     * Sets mask value for column mask.
     *
     * @param column column of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setColumnMask(int column, boolean value) {
        int rows = getRows();
        for (int row = 0; row < rows; row++)  setMask(row, column, value);
    }

    /**
     * Prints mask in row and column format.
     *
     */
    public void print() {
        int rows = getRows();
        int columns = getColumns();
        for (int row = 0; row < rows; row++) {
            System.out.print("[");
            for (int column = 0; column < columns; column++) {
                System.out.print((isMasked(row, column) ? 1 : 0));
                if (column < columns - 1) System.out.print(" ");
            }
            System.out.println("]");
        }
    }

    /**
     * Prints size (rows x columns) of mask.
     *
     */
    public void printSize() {
        System.out.println("Mask size: " + getRows() + "x" + getColumns());
    }

}
