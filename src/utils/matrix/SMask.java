package utils.matrix;

import java.util.HashMap;
import java.util.Stack;

/**
 * Implements sparse mask to mask sparse matrices.
 *
 */
public class SMask extends Mask {

    /**
     * Defines number of rows in mask.
     *
     */
    private final int rows;

    /**
     * Defines number of columns in mask.
     *
     */
    private final int cols;

    /**
     * Hash map to store mask information.
     *
     */
    private HashMap<Integer, Boolean> mask = new HashMap<>();

    /**
     * Hash map to store row mask information.
     *
     */
    private HashMap<Integer, Boolean> rowMask = new HashMap<>();

    /**
     * Hash map to store column mask information.
     *
     */
    private HashMap<Integer, Boolean> colMask = new HashMap<>();

    /**
     * Stack to store masks.
     */
    private Stack<HashMap<Integer, Boolean>> maskStack = new Stack<>();

    /**
     * Stack to store row masks.
     *
     */
    private Stack<HashMap<Integer, Boolean>> rowMaskStack = new Stack<>();

    /**
     * Stack to store column masks.
     *
     */
    private Stack<HashMap<Integer, Boolean>> colMaskStack = new Stack<>();

    /**
     * Constructor for sparse mask.
     *
     * @param rows defines number of rows in mask.
     * @param cols defines number of columns in mask.
     */
    public SMask(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
    }

    /**
     * Constructor for sparse mask.
     *
     * @param rows defines number of rows in mask.
     * @param cols defines number of columns in mask.
     * @param data mask data.
     */
    public SMask(int rows, int cols, HashMap<Integer, Boolean> data) {
        this.rows = rows;
        this.cols = cols;
        mask.putAll(data);
    }

    /**
     * Constructor for sparse mask.
     *
     * @param rows defines number of rows in mask.
     * @param cols defines number of columns in mask.
     * @param data mask data.
     * @param rowData row mask data.
     * @param colData column mask data.
     */
    public SMask(int rows, int cols, HashMap<Integer, Boolean> data, HashMap<Integer, Boolean> rowData, HashMap<Integer, Boolean> colData) {
        this.rows = rows;
        this.cols = cols;
        mask.putAll(data);
        rowMask.putAll(rowData);
        colMask.putAll(colData);
    }

    /**
     * Constructor for sparse mask.
     *
     * @param rows defines number of rows in mask.
     * @param cols defines number of columns in mask.
     * @param data mask data.
     * @param rowData row mask data.
     * @param colData column mask data.
     * @param t if true mask if transposed otherwise false.
     * @param proba probability of masking.
     */
    public SMask(int rows, int cols, HashMap<Integer, Boolean> data, HashMap<Integer, Boolean> rowData, HashMap<Integer, Boolean> colData, boolean t, double proba) {
        this.rows = rows;
        this.cols = cols;
        mask.putAll(data);
        rowMask.putAll(rowData);
        colMask.putAll(colData);
        this.t = t;
        this.proba = proba;
    }

    /**
     * Retrieves copy of mask.
     *
     * @return copy of mask.
     */
    public Mask getCopy() {
        return new SMask(rows, cols, mask, rowMask, colMask, t, proba);
    }

    /**
     * Returns size (rows * columns) of mask
     *
     * @return size of mask.
     */
    public int getSize() {
        return rows * cols;
    }

    /**
     * Returns number of rows in mask.
     *
     * @return number of rows in mask.
     */
    public int getRows() {
        return !t ? rows : cols;
    }

    /**
     * Returns number of columns in mask.
     *
     * @return number of columns in mask.
     */
    public int getCols() {
        return !t ? cols : rows;
    }

    /**
     * Internal function used to set masking of specific row and column.
     *
     * @param row row of value to be set.
     * @param col column of value to be set.
     * @param value defines if specific row and column is masked (true) or not (false).
     */
    public void setMask(int row, int col, boolean value) {
        int curRow = !t ? row : col;
        int curCol = !t ? col : row;
        mask.put(curRow * cols + curCol, value);
    }

    /**
     * Internal function used to get masking of specific row and column.
     *
     * @param row row of value to be returned.
     * @param col column of value to be returned.
     * @return if specific row and column if masked (true) or not (false).
     */
    public boolean getMask(int row, int col) {
        if (mask == null) return false;
        int curRow = !t ? row : col;
        int curCol = !t ? col : row;
        return mask.getOrDefault(curRow * cols + curCol, false);
    }

    /**
     * Clears and removes mask.
     *
     */
    public void clear() {
        mask = new HashMap<>();
        rowMask = new HashMap<>();
        colMask = new HashMap<>();
        maskStack = new Stack<>();
        rowMaskStack = new Stack<>();
        colMaskStack = new Stack<>();
    }

    /**
     * Pushes current mask into stack and optionally creates new mask.<br>
     * Useful in operations where sequence of operations and taken between matrices.<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     */
    public void stackMask(boolean reset) {
        maskStack.push(mask);
        if (reset) mask = new HashMap<>();
    }

    /**
     * Pops mask from mask stack.
     *
     * @throws MatrixException throws exception if mask stack is empty.
     */
    public void unstackMask() throws MatrixException {
        if (maskStack.isEmpty()) throw new MatrixException("Mask stack is empty.");
        mask = maskStack.pop();
    }

    /**
     * Returns size of a mask stack.
     *
     * @return Size of mask stack.
     */
    public int maskStackSize() {
        return maskStack.size();
    }

    /**
     * Clears mask stack.
     *
     */
    public void clearMaskStack() {
        maskStack = new Stack<>();
    }

    /**
     * Pushes current row mask into stack and optionally creates new mask.<br>
     * Useful in operations where sequence of operations and taken between matrices.<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     */
    public void stackRowMask(boolean reset) {
        rowMaskStack.push(rowMask);
        if (reset) rowMask = new HashMap<>();
    }

    /**
     * Pops row mask from mask stack.
     *
     * @throws MatrixException throws exception if row mask stack is empty.
     */
    public void unstackRowMask() throws MatrixException {
        if (rowMaskStack.isEmpty()) throw new MatrixException("Row mask stack is empty.");
        rowMask = rowMaskStack.pop();
    }

    /**
     * Returns size of a row mask stack.
     *
     * @return size of row mask stack.
     */
    public int rowMaskStackSize() {
        return rowMaskStack.size();
    }

    /**
     * Clears row mask stack.
     *
     */
    public void clearRowMaskStack() {
        rowMaskStack = new Stack<>();
    }

    /**
     * Sets mask value for row mask.
     *
     * @param row row of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setRowMask(int row, boolean value) {
        if (!t) rowMask.put(row, value);
        else colMask.put(row, value);
    }

    /**
     * Returns mask value for row mask.
     *
     * @param row row of mask to be returned.
     * @return true if row mask is set otherwise false.
     */
    public boolean getRowMask(int row) {
        if (mask == null) return false;
        return !t ? rowMask.getOrDefault(row, false) : colMask.getOrDefault(row, false);
    }

    /**
     * Sets mask value for column mask.
     *
     * @param col column of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     */
    public void setColMask(int col, boolean value) {
        if (!t) colMask.put(col, value);
        else rowMask.put(col, value);
    }

    /**
     * Returns mask value for column mask.
     *
     * @param col column of mask to be returned.
     * @return true if row mask is set otherwise false.
     */
    public boolean getColMask(int col) {
        if (mask == null) return false;
        return !t ? colMask.getOrDefault(col, false) : rowMask.getOrDefault(col, false);
    }

    /**
     * Pushes current column mask into stack and optionally creates new mask.<br>
     * Useful in operations where sequence of operations and taken between matrices.<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     */
    public void stackColMask(boolean reset) {
        colMaskStack.push(colMask);
        if (reset) colMask = new HashMap<>();
    }

    /**
     * Pops column mask from mask stack.
     *
     * @throws MatrixException throws exception if column mask stack is empty.
     */
    public void unstackColMask() throws MatrixException {
        if (colMaskStack.isEmpty()) throw new MatrixException("Column mask stack is empty.");
        colMask = colMaskStack.pop();
    }

    /**
     * Returns size of a column mask stack.
     *
     * @return size of column mask stack.
     */
    public int colMaskStackSize() {
        return colMaskStack.size();
    }

    /**
     * Clears column mask stack.
     *
     */
    public void clearColMaskStack() {
        colMaskStack = new Stack<>();
    }

}
