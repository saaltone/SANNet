/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.matrix;

import java.io.Serializable;
import java.util.Random;

/**
 * Implements abstract class for masking of matrices.
 *
 */
public abstract class Mask implements Cloneable, Serializable {

    private static final long serialVersionUID = 6859732790552024130L;

    /**
     * Defines if mask is transposed (true) or not (false).
     *
     */
    boolean isTransposed;

    /**
     * Bernoulli probability for selecting if entry (row, column) is masked or not.
     *
     */
    double probability = 0;

    /**
     * Random function for mask class.
     *
     */
    private final Random random = new Random();

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
     * Creates new mask with object full copy of this mask.
     *
     * @return newly created reference mask.
     */
    public Mask copy() {
        return getCopy();
    }

    /**
     * Retrieves copy of mask.
     *
     * @return copy of mask.
     */
    public abstract Mask getCopy();

    /**
     * Transposes mask.
     *
     * @return reference to this mask but with transposed that is flipped rows and columns.
     */
    public Mask transpose() {
        try {
            // Make shallow copy of mask leaving references internal objects which are shared.
            Mask clone = (Mask)clone();
            clone.isTransposed = !clone.isTransposed; // transpose
            return clone;
        } catch (CloneNotSupportedException exception) {
            System.out.println("Mask cloning failed");
        }
        return null;
    }

    /**
     * Checks if mask is transposed.
     *
     * @return true is mask is transposed otherwise false.
     */
    public boolean isTransposed() {
        return isTransposed;
    }

    /**
     * Returns size (rows * columns) of mask.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @return size of matrix.
     */
    public abstract int size();

    /**
     * Returns number of rows in mask.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @return number of rows in mask.
     */
    public abstract int getRows();

    /**
     * Returns number of columns in mask.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @return number of columns in mask.
     */
    public abstract int getColumns();

    /**
     * Clears mask.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     */
    protected abstract void clear();

    /**
     * Checks if mask is set at specific row and / or column
     *
     * @param row row to be checked.
     * @param column column to be checked.
     * @return result of mask check.
     */
    public boolean isMasked(int row, int column) {
        return getRowMask(row) || getColumnMask(column) || getMask(row, column);
    }

    /**
     * Sets bernoulli probability to mask specific element.
     *
     * @param probability masking probability between 0 (0%) and 1 (100%).
     * @throws MatrixException throws exception if masking probability is not between 0 and 1.
     */
    public void setMaskProbability(double probability) throws MatrixException {
        if (probability < 0 || probability > 1) throw new MatrixException("Masking probability must be between 0 and 1.");
        this.probability = probability;
    }

    /**
     * Returns current bernoulli masking probability.
     *
     * @return masking probability.
     */
    public double getMaskProbability() {
        return probability;
    }

    /**
     * Pushes current mask into stack and optionally creates new mask.<br>
     * Useful in operations where sequence of operations and taken between matrices.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     */
    public abstract void stackMask(boolean reset);

    /**
     * Pops mask from mask stack.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @throws MatrixException throws exception if mask stack is empty or matrix is not set.
     */
    public abstract void unstackMask() throws MatrixException;

    /**
     * Returns size of a mask stack.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @return size of mask stack.
     */
    public abstract int maskStackSize();

    /**
     * Clears mask stack.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     */
    public abstract void clearMaskStack();

    /**
     * Returns true with defined masking probability of this matrix.
     *
     * @return true with defined masking probability.
     */
    private boolean isMaskedByProbability() {
        return random.nextDouble() > probability;
    }

    /**
     * Sets mask for element at specific row and column.
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @param row row of mask to be set.
     * @param column column of mask to be set.
     * @param value sets mask if true otherwise unsets mask.
     */
    public abstract void setMask(int row, int column, boolean value);

    /**
     * Returns mask for element at specific row and column.
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @param row row of mask to be returned.
     * @param column column of mask to be returned.
     * @return true if mask is set otherwise false.
     */
    public abstract boolean getMask(int row, int column);

    /**
     * Sets masking for this matrix with given bernoulli probability.
     *
     */
    public void maskByProbability() {
        for (int row = 0; row < getRows(); row++) {
            for (int column = 0; column < getColumns(); column++) {
                setMask(row, column, isMaskedByProbability());
            }
        }
    }

    /**
     * Pushes current row mask into stack and optionally creates new mask for this matrix.<br>
     * Useful in operations where sequence of operations and taken between this matrix and other matrices.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     */
    public abstract void stackRowMask(boolean reset);

    /**
     * Pops row mask from mask stack.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @throws MatrixException throws exception if row mask stack is empty or row matrix is not set.
     */
    public abstract void unstackRowMask() throws MatrixException;

    /**
     * Returns size of a row mask stack.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @return size of row mask stack.
     */
    public abstract int rowMaskStackSize();

    /**
     * Clears row mask stack.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     */
    public abstract void clearRowMaskStack();

    /**
     * Sets mask value for row mask.
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @param row row of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public abstract void setRowMask(int row, boolean value);

    /**
     * Returns mask value for row mask.
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @param row row of mask to be returned.
     * @return true if row mask is set otherwise false.
     */
    public abstract boolean getRowMask(int row);

    /**
     * Sets row masking for this matrix with given bernoulli probability.
     *
     */
    public void maskRowByProbability() {
        for (int row = 0; row < getRows(); row++) {
            setRowMask(row, isMaskedByProbability());
        }
    }

    /**
     * Sets mask value for column mask.
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @param column column of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public abstract void setColumnMask(int column, boolean value);

    /**
     * Sets column masking for this matrix with given bernoulli probability.
     *
     */
    public void maskColumnByProbability() {
        for (int column = 0; column < getColumns(); column++) {
            setColumnMask(column, isMaskedByProbability());
        }
    }

    /**
     * Returns mask value for column mask.
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @param col column of mask to be returned.
     * @return true if row mask is set otherwise false.
     */
    public abstract boolean getColumnMask(int col);

    /**
     * Pushes current column mask into stack and optionally creates new mask for this matrix.<br>
     * Useful in operations where sequence of operations and taken between this matrix and other matrices.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     */
    public abstract void stackColumnMask(boolean reset);

    /**
     * Pops column mask from mask stack.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @throws MatrixException throws exception if mask is not set or column mask stack is empty.
     */
    public abstract void unstackColumnMask() throws MatrixException;

    /**
     * Returns size of a column mask stack.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @return size of column mask stack.
     */
    public abstract int columnMaskStackSize();

    /**
     * Clears column mask stack.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     */
    public abstract void clearColumnMaskStack();

    /**
     * Prints mask in row and column format.
     *
     */
    public void print() {
        for (int row = 0; row < getRows(); row++) {
            System.out.print("[");
            for (int column = 0; column < getColumns(); column++) {
                System.out.print((isMasked(row, column) ? 1 : 0));
                if (column < getColumns() - 1) System.out.print(" ");
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
