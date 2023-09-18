/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Implements sparse matrix.<br>
 * Sparse matrix optimizes matrix memory usage by storing only non-zero values.<br>
 * This matrix type is useful when input sample is expected to contain mostly zero values.<br>
 *
 */
public class SMatrix extends ComputableMatrix {

    /**
     * Matrix data structure as hash map.
     *
     */
    private HashMap<Integer, Double> matrix = new HashMap<>();

    /**
     * Constructor for scalar matrix (size 1x1).
     *
     * @param scalarValue value for matrix.
     */
    public SMatrix(double scalarValue) {
        super(1, 1, 1, true);
        setValue(0, 0, 0, scalarValue);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth depth of matrix.
     * @param mask defines mask of matrix.
     * @throws MatrixException throws exception if new mask dimensions or mask type are not matching with this mask.
     */
    public SMatrix(int rows, int columns, int depth, Mask mask) throws MatrixException {
        this(rows, columns, depth);
        if (mask != null) setMask(mask);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth depth of matrix.
     */
    public SMatrix(int rows, int columns, int depth) {
        super(rows, columns, depth);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     */
    public SMatrix(int rows, int columns, int depth, boolean isScalar) {
        super(rows, columns, depth, isScalar);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth depth of matrix.
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     */
    public SMatrix(int rows, int columns, int depth, Initialization initialization, int inputs, int outputs) {
        this(rows, columns, depth);
        initialize(initialization, inputs, outputs);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     */
    public SMatrix(int rows, int columns, int depth, boolean isScalar, Initialization initialization, int inputs, int outputs) {
        this(rows, columns, depth, isScalar);
        initialize(initialization, inputs, outputs);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth depth of matrix.
     * @param initialization type of initialization defined in class Init.
     */
    public SMatrix(int rows, int columns, int depth, Initialization initialization) {
        this(rows, columns, depth);
        initialize(initialization);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param initialization type of initialization defined in class Init.
     */
    public SMatrix(int rows, int columns, int depth, boolean isScalar, Initialization initialization) {
        this(rows, columns, depth, isScalar);
        initialize(initialization);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth depth of matrix.
     * @param initializer initializer.
     */
    public SMatrix(int rows, int columns, int depth, Matrix.Initializer initializer) {
        this(rows, columns, depth);
        initialize(initializer);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param initializer initializer.
     */
    public SMatrix(int rows, int columns, int depth, boolean isScalar, Matrix.Initializer initializer) {
        this(rows, columns, depth, isScalar);
        initialize(initializer);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth depth of matrix.
     * @param data clones matrix data from given matrix data.
     */
    public SMatrix(int rows, int columns, int depth, HashMap<Integer, Double> data) {
        this(rows, columns, depth);
        for (Map.Entry<Integer, Double> entry : data.entrySet()) {
            int index = entry.getKey();
            double value = entry.getValue();
            matrix.put(index, value);
        }
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param data clones matrix data from given matrix data.
     */
    public SMatrix(int rows, int columns, int depth, boolean isScalar, HashMap<Integer, Double> data) {
        this(rows, columns, depth, isScalar);
        for (Map.Entry<Integer, Double> entry : data.entrySet()) {
            int index = entry.getKey();
            double value = entry.getValue();
            matrix.put(index, value);
        }
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth depth of matrix.
     * @param data matrix data.
     * @param isTransposed if true matrix is transposed and if false not transposed.
     */
    public SMatrix(int rows, int columns, int depth, HashMap<Integer, Double> data, boolean isTransposed) {
        super(rows, columns, depth, false, isTransposed);
        matrix.putAll(data);
    }

    /**
     * Constructor for sparse matrix.
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param isTransposed if true matrix is transposed and if false not transposed.
     * @param data matrix data.
     */
    public SMatrix(int rows, int columns, int depth, boolean isScalar, boolean isTransposed, HashMap<Integer, Double> data) {
        super(rows, columns, depth, isScalar, isTransposed);
        matrix.putAll(data);
    }

    /**
     * Constructor for sparse matrix.
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param isTransposed if true matrix is transposed and if false not transposed.
     * @param canBeSliced if true matrix can be slides otherwise cannot be sliced.
     * @param data matrix data.
     */
    public SMatrix(int rows, int columns, int depth, boolean isScalar, boolean isTransposed, boolean canBeSliced, HashMap<Integer, Double> data) {
        super(rows, columns, depth, isScalar, isTransposed, canBeSliced);
        matrix.putAll(data);
    }

    /**
     * Creates new matrix with object full copy of this matrix.
     *
     * @return newly created copy of matrix.
     * @throws MatrixException throws exception if mask is not set or cloning of matrix fails.
     */
    public Matrix copy() throws MatrixException {
        Matrix newMatrix = new SMatrix(getPureRows(), getPureColumns(), getPureDepth(), isScalar(), isTransposed(), matrix);
        super.setParameters(newMatrix);
        return newMatrix;
    }

    /**
     * Creates new matrix with object full copy of this matrix.
     *
     * @param canBeSliced if true matrix can be slides otherwise cannot be sliced.
     * @return newly created copy of matrix.
     * @throws MatrixException throws exception if mask is not set or cloning of matrix fails.
     */
    public Matrix copy(boolean canBeSliced) throws MatrixException {
        Matrix newMatrix = new SMatrix(getPureRows(), getPureColumns(), getPureDepth(), isScalar(), isTransposed(), canBeSliced, matrix);
        super.setParameters(newMatrix);
        return newMatrix;
    }

    /**
     * Redimensions matrix assuming new dimensions are matching.
     *
     * @param newRows new row size
     * @param newColumns new column size
     * @param newDepth new depth size.
     * @return redimensioned matrix.
     * @throws MatrixException throws exception if redimensioning fails.
     */
    public Matrix redimension(int newRows, int newColumns, int newDepth) throws MatrixException {
        if (newRows * newColumns * newDepth != getPureRows() * getPureColumns() * getPureDepth()) throw new MatrixException("Matrix of size: " + getPureRows() + "x" + getPureColumns() + "x" + getPureDepth() + " cannot be redimensioned to size: " + newRows + "x" + newColumns + "x" + newDepth);
        Matrix newMatrix = new SMatrix(newRows, newColumns, newDepth, isScalar(), isTransposed(), matrix);
        super.setParameters(newMatrix);
        return newMatrix;
    }

    /**
     * Redimensions matrix assuming new dimensions are matching.
     *
     * @param newRows new row size
     * @param newColumns new column size
     * @param newDepth new depth size.
     * @return redimensioned matrix.
     * @throws MatrixException throws exception if redimensioning fails.
     */
    public Matrix redimension(int newRows, int newColumns, int newDepth, boolean copyData) throws MatrixException {
        return redimension(newRows, newColumns, newDepth);
    }

    /**
     * Transposes matrix.
     *
     * @return transposed matrix.
     * @throws MatrixException throws exception if cloning of mask fails.
     */
    protected Matrix applyTranspose() throws MatrixException {
        Matrix newMatrix = new SMatrix(getPureRows(), getPureColumns(), getPureDepth(), isScalar(), true, matrix);
        super.setParameters(newMatrix);
        return newMatrix;
    }

    /**
     * Returns sub-matrices within matrix.
     *
     * @return sub-matrices within matrix.
     */
    public ArrayList<Matrix> getSubMatrices() {
        ArrayList<Matrix> matrices = new ArrayList<>();
        matrices.add(this);
        return matrices;
    }

    /**
     * Resets matrix leaving dimensions same.
     *
     */
    public void resetMatrix() {
        matrix = new HashMap<>();
    }

    /**
     * Sets value of matrix at specific row and column.
     *
     * @param row row of value to be set.
     * @param column column of value to be set.
     * @param depth depth of value to be set.
     * @param value new value to be set.
     */
    public void setValue(int row, int column, int depth, double value) {
        if (value != 0) matrix.put(getArrayIndex(row, column, depth), value);
    }

    /**
     * Returns value of matrix at specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @param depth depth of value to be returned.
     * @return value of row and column.
     */
    public double getValue(int row, int column, int depth) {
        return matrix.getOrDefault(getArrayIndex(row, column, depth), (double)0);
    }

    /**
     * Returns matrix of given size (rows x columns)
     *
     * @param rows rows
     * @param columns columns
     * @param depth depth
     * @return new matrix
     * @throws MatrixException throws exception if new mask dimensions or mask type are not matching with this mask.
     */
    public Matrix getNewMatrix(int rows, int columns, int depth) throws MatrixException {
        return forceDMatrix ? new DMatrix(rows, columns, depth, getMask() != null ? new DMask(rows, columns, depth) : null) : new SMatrix(rows, columns, depth, getMask() != null ? getNewMask(rows, columns, depth) : null);
    }

    /**
     * Returns constant matrix
     *
     * @param constant constant
     * @return new matrix
     */
    protected Matrix getNewMatrix(double constant) {
        return forceDMatrix ? new DMatrix(constant) : new SMatrix(constant);
    }

    /**
     * Returns new mask for this matrix.
     *
     * @return mask of this matrix.
     */
    protected Mask getNewMask() {
        return new SMask(getTotalRows(), getTotalColumns(), getTotalDepth());
    }

    /**
     * Returns new mask for this matrix.
     *
     * @param rows rows
     * @param columns columns
     * @param depth depth
     * @return mask of this matrix.
     */
    protected Mask getNewMask(int rows, int columns, int depth) {
        return new SMask(rows, columns, depth);
    }

    /**
     * Return one-hot encoded column vector.
     *
     * @param size size of vector
     * @param position position of one-hot encoded value
     * @return one-hot encoded vector.
     * @throws MatrixException throws exception if position of one-hot encoded value exceeds vector size.
     */
    public static Matrix getOneHotVector(int size, int position) throws MatrixException {
        return getOneHotVector(size, position, true);
    }

    /**
     * Return one-hot encoded vector.
     *
     * @param size size of vector
     * @param position position of one-hot encoded value
     * @param asColumnVector if true one-hot vector is column vector otherwise row vector
     * @return one-hot encoded vector.
     * @throws MatrixException throws exception if position of one-hot encoded value exceeds vector size.
     */
    public static Matrix getOneHotVector(int size, int position, boolean asColumnVector) throws MatrixException {
        if (position > size - 1) throw new MatrixException("Position " + position + " cannot exceed vector size " + size);
        Matrix oneHotVector = new SMatrix(asColumnVector ? size : 1, asColumnVector ? 1 : size, 1);
        oneHotVector.setValue(asColumnVector ? position : 0, asColumnVector ? 0 : position, 0, 1);
        return oneHotVector;
    }

}
