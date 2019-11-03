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
     */
    boolean t;

    /**
     * Bernoulli probability for selecting if entry (row, column) is masked or not.
     */
    double proba = 0;

    /**
     * Random function for mask class.
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
    public Mask T() {
        try {
            // Make shallow copy of mask leaving references internal objects which are shared.
            Mask clone = (Mask)clone();
            clone.t = !clone.t; // transpose
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
    public boolean isT() {
        return t;
    }

    /**
     * Returns size (rows * columns) of mask.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @return size of matrix.
     */
    public abstract int getSize();

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
    public abstract int getCols();

    /**
     * Clears mask.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     */
    protected abstract void clear();

    /**
     * Checks if mask is set at specific row and / or col
     *
     * @param row row to be checked.
     * @param col col to be checked.
     * @return result of mask check.
     */
    public boolean isMasked(int row, int col) {
        return getRowMask(row) || getColMask(col) || getMask(row, col);
    }

    /**
     * Sets bernoulli probability to mask specific element.
     *
     * @param proba masking probability between 0 (0%) and 1 (100%).
     * @throws MatrixException throws exception if masking probability is not between 0 and 1.
     */
    public void setMaskProba(double proba) throws MatrixException {
        if (proba < 0 || proba > 1) throw new MatrixException("Masking probability must be between 0 and 1.");
        this.proba = proba;
    }

    /**
     * Returns current bernoulli masking probability.
     *
     * @return masking probability.
     */
    public double getMaskProba() {
        return proba;
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
     * Returns true with probability of proba. Proba is masking probability of this matrix.
     *
     * @return true with probability of proba.
     */
    private boolean maskByProbability() {
        return random.nextDouble() > proba;
    }

    /**
     * Sets mask for element at specific row and column.
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @param row row of mask to be set.
     * @param col column of mask to be set.
     * @param value sets mask if true otherwise unsets mask.
     */
    public abstract void setMask(int row, int col, boolean value);

    /**
     * Returns mask for element at specific row and column.
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @param row row of mask to be returned.
     * @param col column of mask to be returned.
     * @return true if mask is set otherwise false.
     */
    public abstract boolean getMask(int row, int col);

    /**
     * Sets masking for this matrix with given bernoulli probability proba.
     *
     */
    public void maskByProba() {
        for (int row = 0; row < getRows(); row++) {
            for (int col = 0; col < getCols(); col++) {
                // Mask out (set value as true) true by probability of proba
                if (maskByProbability()) setMask(row, col, true);
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
     * Sets row masking for this matrix with given bernoulli probability proba.
     *
     */
    public void maskRowByProba() {
        for (int row = 0; row < getRows(); row++) {
            // Mask out (set value as true) by probability of proba
            if (maskByProbability()) setRowMask(row, true);
        }
    }

    /**
     * Sets mask value for column mask.
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @param col column of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public abstract void setColMask(int col, boolean value);

    /**
     * Sets column masking for this matrix with given bernoulli probability proba.
     *
     */
    public void maskColByProba() {
        for (int col = 0; col < getCols(); col++) {
            // Mask out (set value as true) true by probability of proba
            if (maskByProbability()) setColMask(col, true);
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
    public abstract boolean getColMask(int col);

    /**
     * Pushes current column mask into stack and optionally creates new mask for this matrix.<br>
     * Useful in operations where sequence of operations and taken between this matrix and other matrices.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     */
    public abstract void stackColMask(boolean reset);

    /**
     * Pops column mask from mask stack.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @throws MatrixException throws exception if mask is not set or column mask stack is empty.
     */
    public abstract void unstackColMask() throws MatrixException;

    /**
     * Returns size of a column mask stack.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     * @return size of column mask stack.
     */
    public abstract int colMaskStackSize();

    /**
     * Clears column mask stack.<br>
     * Abstract function to be implemented by underlying mask data structure class implementation.<br>
     * This is typically dense mask (DMask class) or sparse mask (SMask class).<br>
     *
     */
    public abstract void clearColMaskStack();

}
