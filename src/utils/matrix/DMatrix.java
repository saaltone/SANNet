/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Implements dense matrix.<br>
 * Dense matrix assumes full array data structure including storage of zero values.<br>
 *
 */
public class DMatrix extends ComputableMatrix {

    /**
     * Defines matrix data structure using 1-dimensional row column array.
     *
     */
    private double[] matrix;

    /**
     * Constructor for scalar matrix (size 1x1x1).
     *
     * @param scalarValue value for matrix.
     */
    public DMatrix(double scalarValue) {
        super(1, 1, 1,true);
        matrix = new double[1];
        matrix[0] = scalarValue;
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param mask defines mask of matrix.
     * @throws MatrixException throws exception if new mask dimensions or mask type are not matching with this mask.
     */
    public DMatrix(int rows, int columns, int depth, Mask mask) throws MatrixException {
        this(rows, columns, depth);
        if (mask != null) setMask(mask);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     */
    public DMatrix(int rows, int columns, int depth) {
        super(rows, columns, depth);
        matrix = new double[rows * columns * depth];
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     */
    public DMatrix(int rows, int columns, int depth, boolean isScalar) {
        super(rows, columns, depth, isScalar);
        matrix = new double[rows * columns * depth];
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     */
    public DMatrix(int rows, int columns, int depth, Initialization initialization, int inputs, int outputs) {
        this(rows, columns, depth);
        initialize(initialization, inputs, outputs);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     */
    public DMatrix(int rows, int columns, int depth, boolean isScalar, Initialization initialization, int inputs, int outputs) {
        this(rows, columns, depth, isScalar);
        initialize(initialization, inputs, outputs);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param initialization type of initialization defined in class Init.
     */
    public DMatrix(int rows, int columns, int depth, Initialization initialization) {
        this(rows, columns, depth);
        initialize(initialization);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param initialization type of initialization defined in class Init.
     */
    public DMatrix(int rows, int columns, int depth, boolean isScalar, Initialization initialization) {
        this(rows, columns, depth, isScalar);
        initialize(initialization);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param initializer initializer.
     */
    public DMatrix(int rows, int columns, int depth, Matrix.Initializer initializer) {
        this(rows, columns, depth);
        initialize(initializer);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param initializer initializer.
     */
    public DMatrix(int rows, int columns, int depth, boolean isScalar, Matrix.Initializer initializer) {
        this(rows, columns, depth, isScalar);
        initialize(initializer);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param data clones matrix data from given matrix data.
     */
    public DMatrix(int rows, int columns, int depth, double[] data) {
        super(rows, columns, depth);
        matrix = data.clone();
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param data clones matrix data from given matrix data.
     * @param isScalar true if matrix is scalar (size 1x1).
     */
    public DMatrix(int rows, int columns, int depth, double[] data, boolean isScalar) {
        super(rows, columns, depth, isScalar);
        matrix = data.clone();
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param data matrix data.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param isTransposed if true matrix is transposed and if false not transposed.
     */
    public DMatrix(int rows, int columns, int depth, double[] data, boolean isScalar, boolean isTransposed) {
        super(rows, columns, depth, isScalar, isTransposed);
        matrix = data;
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param data matrix data.
     * @param copyData if true matrix data is copied and if false referenced.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param isTransposed if true matrix is transposed and if false not transposed.
     */
    public DMatrix(int rows, int columns, int depth, double[] data, boolean copyData, boolean isScalar, boolean isTransposed) {
        super(rows, columns, depth, isScalar, isTransposed);
        matrix = copyData ? data.clone() : data;
    }

    /**
     * Constructor for dense matrix.
     *
     * @param other matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public DMatrix(Matrix other) throws MatrixException {
        this(other.getRows(), other.getColumns(), other.getDepth());
        setEqualTo(other);
    }

    /**
     * Creates new matrix with object full copy of this matrix.
     *
     * @return newly created reference matrix.
     * @throws MatrixException throws exception if mask is not set or cloning of matrix fails.
     */
    public Matrix copy() throws MatrixException {
        Matrix newMatrix = new DMatrix(getPureRows(), getPureColumns(), getPureDepth(), matrix, isScalar(), isTransposed());
        super.setParameters(newMatrix);
        return newMatrix;
    }

    /**
     * Transposes matrix.
     *
     * @return transposed matrix.
     * @throws MatrixException throws exception if cloning of mask fails.
     */
    protected Matrix applyTranspose() throws MatrixException {
        Matrix newMatrix = new DMatrix(getPureRows(), getPureColumns(), getPureDepth(), matrix, isScalar(), true);
        super.setParameters(newMatrix);
        return newMatrix;
    }

    /**
     * Checks if data of other matrix is equal to data of this matrix
     *
     * @param other matrix to be compared.
     * @return true is data of this and other matrix are equal otherwise false.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public boolean equals(Matrix other) throws MatrixException {
        if (other instanceof DMatrix otherDMatrix) {
            if (other.getRows() != getRows() || other.getColumns() != getColumns() || other.getDepth() != getDepth()) {
                throw new MatrixException("Incompatible target matrix size: " + other.getRows() + "x" + other.getColumns() + "x" + other.getDepth());
            }
            return otherDMatrix.isEqual(matrix);
        }
        else return super.equals(other);
    }

    /**
     * Checks if matrix data equals to data of this matrix.
     *
     * @return true if matrix data and data of this matrix are equal otherwise returns false.
     */
    private boolean isEqual(double[] matrixData) {
        return Arrays.equals(matrix, matrixData);
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
        matrix = new double[getPureRows() * getPureColumns() * getPureDepth()];
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
        matrix[getArrayIndex(row, column, depth)] = value;
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
        return matrix[getArrayIndex(row, column, depth)];
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
        return new DMatrix(rows, columns, depth, getMask() != null ? getNewMask() : null);
    }

    /**
     * Returns constant matrix
     *
     * @param constant constant
     * @return new matrix
     */
    protected Matrix getNewMatrix(double constant) {
        return new DMatrix(constant);
    }

    /**
     * Returns new mask for this matrix.
     *
     * @return mask of this matrix.
     */
    protected Mask getNewMask() {
        return new DMask(getTotalRows(), getTotalColumns(), getTotalDepth());
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
        Matrix oneHotVector = new DMatrix(asColumnVector ? size : 1, asColumnVector ? 1 : size, 1);
        oneHotVector.setValue(asColumnVector ? position : 0, asColumnVector ? 0 : position, 0, 1);
        return oneHotVector;
    }

}
