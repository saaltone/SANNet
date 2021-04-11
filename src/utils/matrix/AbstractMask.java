package utils.matrix;

import java.io.Serializable;
import java.util.Random;

/**
 * Abstract class that implements common operations for masking.<br>
 *
 */
public abstract class AbstractMask implements Cloneable, Serializable, Mask {

    private static final long serialVersionUID = -4902569287054022460L;

    /**
     * Bernoulli probability for selecting if entry (row, column) is masked or not.
     *
     */
    protected double probability = 0;

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
        for (int row = 0; row < getRows(); row++) {
            if (getRowMask(row)) transposedMask.setColumnMask(row, getRowMask(row));
        }
        for (int column = 0; column < getColumns(); column++) {
            if (getColumnMask(column)) transposedMask.setRowMask(column, getColumnMask(column));
        }
        for (int row = 0; row < getRows(); row++) {
            for (int column = 0; column < getColumns(); column++) {
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
        return getRowMask(row) || getColumnMask(column) || getMask(row, column);
    }

    /**
     * Sets bernoulli probability to mask specific row and column.
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
        for (int row = 0; row < getRows(); row++) {
            for (int column = 0; column < getColumns(); column++) {
                setMask(row, column, isMaskedByProbability());
            }
        }
    }

    /**
     * Sets row masking for this mask with given bernoulli probability.
     *
     */
    public void maskRowByProbability() {
        for (int row = 0; row < getRows(); row++) {
            setRowMask(row, isMaskedByProbability());
        }
    }

    /**
     * Sets column masking for this mask with given bernoulli probability.
     *
     */
    public void maskColumnByProbability() {
        for (int column = 0; column < getColumns(); column++) {
            setColumnMask(column, isMaskedByProbability());
        }
    }

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
