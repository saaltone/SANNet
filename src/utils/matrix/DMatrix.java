/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Matrix class that implements dense matrix.<br>
 * Dense matrix assumes full array data structure including storage of zero values.<br>
 *
 */
public class DMatrix extends ComputableMatrix {

    /**
     * Defines matrix data structure using 2-dimensional row column array.
     *
     */
    private double[][] matrix;

    /**
     * Constructor for scalar matrix (size 1x1).
     *
     * @param scalarValue value for matrix.
     */
    public DMatrix(double scalarValue) {
        super(1, 1,true);
        matrix = new double[1][1];
        matrix[0][0] = scalarValue;
        updateSliceDimensions(0, 0, 0, 0);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     */
    public DMatrix(int rows, int columns) {
        super(rows, columns, false);
        matrix = new double[rows][columns];
        updateSliceDimensions(0, 0, rows - 1, columns - 1);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     */
    public DMatrix(int rows, int columns, Initialization initialization, int inputs, int outputs) {
        this(rows, columns);
        initialize(initialization, inputs, outputs);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     */
    public DMatrix(int rows, int columns, Initialization initialization) {
        this(rows, columns);
        initialize(initialization);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initializer initializer.
     */
    public DMatrix(int rows, int columns, Matrix.Initializer initializer) {
        this(rows, columns);
        initialize(initializer);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param data clones matrix data from given matrix data.
     */
    public DMatrix(double[][] data) {
        super(data.length, data[0].length, false);
        matrix = data.clone();
        updateSliceDimensions(0, 0, data.length - 1, data[0].length - 1);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param data matrix data.
     * @param isTransposed if true matrix is transposed and if false not transposed.
     */
    public DMatrix(double[][] data, boolean isTransposed) {
        super(data.length, data[0].length, false, isTransposed);
        matrix = data;
        updateSliceDimensions(0, 0, data.length - 1, data[0].length - 1);
    }

    /**
     * Constructor for dense matrix.
     *
     * @param data matrix data.
     * @param copyData if true matrix data is copied and if false referenced.
     * @param isTransposed if true matrix is transposed and if false not transposed.
     */
    public DMatrix(double[][] data, boolean copyData, boolean isTransposed) {
        super(data.length, data[0].length, false, isTransposed);
        matrix = copyData ? data.clone() : data;
        updateSliceDimensions(0, 0, data.length - 1, data[0].length - 1);
    }

    /**
     * Creates new matrix with object full copy of this matrix.
     *
     * @return newly created reference matrix.
     * @throws MatrixException throws exception if mask is not set or cloning of matrix fails.
     */
    public Matrix copy() throws MatrixException {
        Matrix newMatrix = new DMatrix(matrix, true, false);
        super.setParameters(newMatrix);
        return newMatrix;
    }

    /**
     * Transposes matrix.
     *
     * @return transposed matrix.
     * @throws MatrixException throws exception if cloning of mask fails.
     */
    public Matrix transpose() throws MatrixException {
        Matrix newMatrix = new DMatrix(matrix, true);
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
            if (other.getRows() != getRows() || other.getColumns() != getColumns()) {
                throw new MatrixException("Incompatible target matrix size: " + other.getRows() + "x" + other.getColumns());
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
    private boolean isEqual(double[][] matrixData) {
        return Arrays.deepEquals(matrix, matrixData);
    }

    /**
     * Makes current matrix data equal to other matrix data.
     *
     * @param other other matrix to be copied as data of this matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void setEqualTo(Matrix other) throws MatrixException {
        if (other instanceof DMatrix otherDMatrix) {
            if (other.getRows() != getRows() || other.getColumns() != getColumns()) {
                throw new MatrixException("Incompatible target matrix size: " + other.getRows() + "x" + other.getColumns());
            }
            otherDMatrix.makeEqual(matrix);
        }
        else super.setEqualTo(other);
    }

    /**
     * Makes matrix data equals to data of this matrix.
     *
     * @param matrixData matrix data
     */
    private void makeEqual(double[][] matrixData) {
        matrix = matrixData.clone();
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
        matrix = new double[matrix.length][matrix[0].length];
    }

    /**
     * Returns new mask for this matrix.
     *
     * @return mask of this matrix.
     */
    protected Mask getNewMask() {
        return new DMask(getTotalRows(), getTotalColumns());
    }

    /**
     * Sets value of matrix at specific row and column.
     *
     * @param row row of value to be set.
     * @param column column of value to be set.
     * @param value new value to be set.
     */
    public void setValue(int row, int column, double value) {
        matrix[isScalar() ? 0 : getSliceStartRow() + (!isTransposed() ? row : column)][isScalar() ? 0 : getSliceStartColumn() + (!isTransposed() ? column : row)] = value;
    }

    /**
     * Returns value of matrix at specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @return value of row and column.
     */
    public double getValue(int row, int column) {
        return matrix[isScalar() ? 0 : getSliceStartRow() + (!isTransposed() ? row : column)][isScalar() ? 0 : getSliceStartColumn() + (!isTransposed() ? column : row)];
    }

    /**
     * Returns matrix of given size (rows x columns)
     *
     * @param rows rows
     * @param columns columns
     * @return new matrix
     */
    protected Matrix getNewMatrix(int rows, int columns) {
        return new DMatrix(rows, columns);
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

}
