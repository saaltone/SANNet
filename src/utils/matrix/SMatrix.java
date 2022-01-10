/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Matrix class that implements sparse matrix.<br>
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
        super(1, 1, true);
        updateSliceDimensions(0, 0, 0, 0);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     */
    public SMatrix(int rows, int columns) {
        super(rows, columns, false);
        updateSliceDimensions(0, 0, getTotalRows() - 1, getTotalColumns() - 1);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     */
    public SMatrix(int rows, int columns, Initialization initialization, int inputs, int outputs) {
        this(rows, columns);
        initialize(initialization, inputs, outputs);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initialization type of initialization defined in class Init.
     */
    public SMatrix(int rows, int columns, Initialization initialization) {
        this(rows, columns);
        initialize(initialization);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param initializer initializer.
     */
    public SMatrix(int rows, int columns, Matrix.Initializer initializer) {
        this(rows, columns);
        initialize(initializer);
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param data clones matrix data from given matrix data.
     */
    public SMatrix(int rows, int columns, HashMap<Integer, Double> data) {
        this(rows, columns);
        for (Integer index : data.keySet()) matrix.put(index, data.get(index));
    }

    /**
     * Constructor for sparse matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param data matrix data.
     * @param referTo if true creates matrix with reference to given matrix data otherwise clones the data.
     */
    public SMatrix(int rows, int columns, HashMap<Integer, Double> data, boolean referTo) {
        this(rows, columns);
        if (referTo) matrix.putAll(data);
        else for (Integer index : data.keySet()) matrix.put(index, data.get(index));
    }

    /**
     * Returns sub-matrices within Matrix.
     *
     * @return sub-matrices within Matrix.
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
     * Returns new mask for this matrix.
     *
     * @return mask of this matrix.
     */
    protected Mask getNewMask() {
        return new SMask(getTotalRows(), getTotalColumns());
    }

    /**
     * Sets value of matrix at specific row and column.
     *
     * @param row row of value to be set.
     * @param column column of value to be set.
     * @param value new value to be set.
     */
    public void setValue(int row, int column, double value) {
        if (value != 0) matrix.put(isScalar() ? 0 : (getSliceStartRow() + row) * getTotalColumns() + (getSliceStartColumn() + column), value);
    }

    /**
     * Returns value of matrix at specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @return value of row and column.
     */
    public double getValue(int row, int column) {
        return matrix.getOrDefault(isScalar() ? 0 : (getSliceStartRow() + row) * getTotalColumns() + (getSliceStartColumn() + column), (double)0);
    }

    /**
     * Returns matrix of given size (rows x columns)
     *
     * @param rows rows
     * @param columns columns
     * @return new matrix
     */
    protected Matrix getNewMatrix(int rows, int columns) {
        return forceDMatrix ? new DMatrix(rows, columns) : new SMatrix(rows, columns);
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

}
