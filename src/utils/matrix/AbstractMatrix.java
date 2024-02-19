/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.matrix;

import utils.configurable.DynamicParamException;
import utils.matrix.operation.*;
import utils.procedure.ProcedureFactory;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Implements abstract matrix that implements common operations for matrices.<br>
 *
 */
public abstract class AbstractMatrix implements Cloneable, Serializable, Matrix {

    @Serial
    private static final long serialVersionUID = 4372639167186260605L;

    /**
     * If true forces creation of DMatrix when new matrix is created out of current matrix.
     *
     */
    protected final static boolean forceDMatrix = true;

    /**
     * If true matrix is treated as scalar (1x1) matrix otherwise as normal matrix.
     *
     */
    private final boolean isScalar;

    /**
     * Number of rows in matrix.
     *
     */
    private final int rows;

    /**
     * Number of columns in matrix.
     *
     */
    private final int columns;

    /**
     * Depth og matrix.
     *
     */
    private final int depth;

    /**
     * Stride size for convolutional and pooling operations.
     *
     */
    private int stride = 1;

    /**
     * Dilation step size for convolutional operations.
     *
     */
    private int dilation = 1;

    /**
     * Filter row size for convolutional and pooling operations.
     *
     */
    private int filterRowSize;

    /**
     * Filter column size for convolutional and pooling operations.
     *
     */
    private int filterColumnSize;

    /**
     * Filter depth for convolutional operation.
     *
     */
    private int filterDepth;

    /**
     * If true convolution is depth separable.
     *
     */
    private boolean isDepthSeparable;

    /**
     * Slice start row.
     *
     */
    private int sliceStartRow;

    /**
     * Slice start column.
     *
     */
    private int sliceStartColumn;

    /**
     * Slice start depth.
     *
     */
    private int sliceStartDepth;

    /**
     * Number of rows in slice.
     *
     */
    private int sliceRows;

    /**
     * Number of columns in slice.
     *
     */
    private int sliceColumns;

    /**
     * Depth of slice.
     *
     */
    private int sliceDepth;

    /**
     * Size of slice (sliceRows * sliceColumns * sliceDepth)
     *
     */
    private int sliceSize;

    /**
     * If true matrix can be slides otherwise cannot be sliced.
     *
     */
    private final boolean canBeSliced;

    /**
     * Name of matrix.
     *
     */
    private String name;

    /**
     * If true matrix is transposed.
     *
     */
    private final boolean isTransposed;

    /**
     * Reference to mask of matrix. If null mask is not used.
     *
     */
    private Mask mask;

    /**
     * Procedure factory reference for matrix.
     * Procedure factory records chain of executed matrix operations enabling dynamic construction of procedure, and it's gradient.
     *
     */
    private transient ProcedureFactory procedureFactory = null;

    /**
     * Random function for matrix class.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     */
    protected AbstractMatrix(int rows, int columns, int depth) {
        this(rows, columns, depth, (rows == 1 && columns == 1 && depth == 1));
    }

    /**
     * Constructor for abstract matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     */
    protected AbstractMatrix(int rows, int columns, int depth, boolean isScalar) {
        this(rows, columns, depth, isScalar, false, false);
    }

    /**
     * Constructor for abstract matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param isTransposed if true matrix is transposed and if false not transposed.
     */
    protected AbstractMatrix(int rows, int columns, int depth, boolean isScalar, boolean isTransposed) {
        this(rows, columns, depth, isScalar, isTransposed, false);
    }

    /**
     * Constructor for abstract matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param isTransposed if true matrix is transposed and if false not transposed.
     * @param canBeSliced if true matrix can be slides otherwise cannot be sliced.
     */
    protected AbstractMatrix(int rows, int columns, int depth, boolean isScalar, boolean isTransposed, boolean canBeSliced) {
        this.rows = rows;
        this.columns = columns;
        this.depth = depth;
        this.isScalar = isScalar && (rows == 1 && columns == 1 && depth == 1);
        this.isTransposed = isTransposed;
        this.canBeSliced = canBeSliced;
        if (canBeSliced()) updateSliceDimensions(0, 0, 0, rows - 1, columns - 1, depth -1);
    }

    /**
     * Sets parameters for matrix.
     *
     * @param matrix matrix.
     */
    protected void setParameters(Matrix matrix) {
        matrix.setName(name);
        matrix.setStride(stride);
        matrix.setDilation(dilation);
        matrix.setFilterRowSize(filterRowSize);
        matrix.setFilterColumnSize(filterColumnSize);
        matrix.setFilterDepth(filterDepth);
    }

    /**
     * Returns true if matrix is scalar otherwise false.
     *
     * @return true if matrix is scalar otherwise false.
     */
    public boolean isScalar() {
        return isScalar;
    }

    /**
     * Returns total number of rows defined for matrix without considering transposing.
     *
     * @return total number of rows defined for matrix without considering transposing.
     */
    protected int getPureRows() {
        return rows;
    }

    /**
     * Returns total number of columns defined for matrix without considering transposing.
     *
     * @return total number of columns defined for matrix without considering transposing.
     */
    protected int getPureColumns() {
        return columns;
    }

    /**
     * Returns total depth defined for matrix without considering transposing.
     *
     * @return total depth defined for matrix without considering transposing.
     */
    protected int getPureDepth() {
        return depth;
    }

    /**
     * Returns total number of rows defined for matrix.
     *
     * @return total number of rows defined for matrix.
     */
    public int getTotalRows() {
        return !isTransposed() ? getPureRows() : getPureColumns();
    }

    /**
     * Returns total number of columns defined for matrix.
     *
     * @return total number of columns defined for matrix.
     */
    public int getTotalColumns() {
        return !isTransposed() ? getPureColumns() : getPureRows();
    }

    /**
     * Returns total depth defined for matrix.
     *
     * @return total depth defined for matrix.
     */
    public int getTotalDepth() {
        return getPureDepth();
    }

    /**
     * Checks if matrix can be sliced.
     *
     * @return true if matrix can be sliced otherwise returns false.
     */
    private boolean canBeSliced() {
        return canBeSliced;
    }

    /**
     * Returns array index based on row and column
     *
     * @param row row
     * @param column column
     * @param depth depth
     * @return array index
     */
    protected int getArrayIndex(int row, int column, int depth) {
        if (isScalar()) return 0;
        else {
            if (canBeSliced()) return (getPureDepth() > 1 ? (getSliceStartDepth() + depth) * getPureRows() * getPureColumns() : 0) + (getPureColumns() > 1 ? (getSliceStartColumn() + (!isTransposed() ? column : row)) * getPureRows() : 0) + getSliceStartRow() + (!isTransposed() ? row : column);
            else return (getPureDepth() > 1 ? (depth) * getPureRows() * getPureColumns() : 0) + (getPureColumns() > 1 ? ((!isTransposed() ? column : row)) * getPureRows() : 0) + (!isTransposed() ? row : column);
        }
    }

    /**
     * Returns value from uniform distribution within -range to +range.
     *
     * @param range range of the distribution.
     * @return random value drawn from the distribution.
     */
    private double uniform(double range) {
        return (2 * random.nextDouble()- 1)  * range;
    }

    /**
     * Returns value from normal distribution defined by standard deviation.
     *
     * @param standardDeviation standard deviation of normal distribution.
     * @return random value drawn from the distribution.
     */
    private double normal(double standardDeviation) {
        return random.nextGaussian() * standardDeviation;
    }

    /**
     * Initializes matrix.
     *
     * @param initialization type of initialization defined in class Init.
     */
    public void initialize(Initialization initialization) {
        initialize(initialization, 0, 0);
    }

    /**
     * Initializes matrix.
     *
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     */
    public void initialize(Initialization initialization, int inputs, int outputs) {
        switch (initialization) {
            case ZERO -> initialize((Initializer & Serializable) (row, column) -> 0);
            case ONE -> initialize((Initializer & Serializable) (row, column) -> 1);
            case RANDOM -> initialize((Initializer & Serializable) (row, column) -> random.nextDouble());
            case IDENTITY -> initialize((Initializer & Serializable) (row, column) -> (row == column) ? 1 : 0);
            case NORMAL_XAVIER -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(2 / (double) (getRows() + getColumns()))));
            case UNIFORM_XAVIER -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(6 / (double) (getRows() + getColumns()))));
            case NORMAL_HE -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(2 / ((double) getRows()))));
            case UNIFORM_HE -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(6 / (double) (getRows()))));
            case NORMAL_LECUN -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(1 / (double) (getRows()))));
            case UNIFORM_LECUN -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(3 / (double) (getRows()))));
            case NORMAL_XAVIER_CONV -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(2 / (double) (outputs + inputs))));
            case UNIFORM_XAVIER_CONV -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(6 / (double) (outputs + inputs))));
            case NORMAL_HE_CONV -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(2 / (double) (outputs))));
            case UNIFORM_HE_CONV -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(6 / (double) (outputs))));
            case NORMAL_LECUN_CONV -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(1 / (double) (outputs))));
            case UNIFORM_LECUN_CONV -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(3 / (double) (outputs))));
            default -> {
            }
        }
    }

    /**
     * Initializes matrix with given initializer operation.
     *
     * @param initializer initializer operation.
     */
    public void initialize(Matrix.Initializer initializer) {
        int rows = getRows();
        int columns = getColumns();
        int totalDepth = getDepth();
        for (int depth = 0; depth < totalDepth; depth++) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    setValue(row, column, depth, initializer.value(row, column));
                }
            }
        }
    }

    /**
     * Initializes matrix with given value.
     *
     * @param value initialization value.
     */
    public void initializeToValue(double value) {
        int rows = getRows();
        int columns = getColumns();
        int totalDepth = getDepth();
        for (int depth = 0; depth < totalDepth; depth++) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    setValue(row, column, depth, value);
                }
            }
        }
    }

    /**
     * Returns slice start row.
     *
     * @return slice start row.
     */
    protected int getSliceStartRow() {
        return canBeSliced() ? sliceStartRow : 0;
    }

    /**
     * Returns slice start column.
     *
     * @return slice start column.
     */
    protected int getSliceStartColumn() {
        return canBeSliced() ? sliceStartColumn : 0;
    }

    /**
     * Returns slice start depth.
     *
     * @return slice start depth.
     */
    protected int getSliceStartDepth() {
        return canBeSliced() ? sliceStartDepth : 0;
    }

    /**
     * Returns number of slice rows.
     *
     * @return number of slice rows.
     */
    protected int getSliceRows() {
        return canBeSliced() ? sliceRows : getPureRows();
    }

    /**
     * Returns number of slice columns.
     *
     * @return number of slice columns.
     */
    protected int getSliceColumns() {
        return canBeSliced() ? sliceColumns : getPureColumns();
    }

    /**
     * Returns depth of slice.
     *
     * @return depth of slice.
     */
    protected int getSliceDepth() {
        return canBeSliced() ? sliceDepth : getPureDepth();
    }

    /**
     * Slices matrix.
     *
     * @param startRow start row of slice.
     * @param startColumn start column of slice.
     * @param startDepth start column of slice.
     * @param endRow  end row of slice.
     * @param endColumn  end column of slice.
     * @param endDepth  end column of slice.
     * @throws MatrixException throws exception if slicing fails.
     */
    public void slice(int startRow, int startColumn, int startDepth, int endRow, int endColumn, int endDepth) throws MatrixException {
        if (!canBeSliced()) throw new MatrixException("Matrix cannot be sliced.");
        if (startRow < 0 || startColumn < 0 || startDepth < 0 || (!isTransposed() ? endRow : endColumn) > getPureRows() -1 || (!isTransposed() ? endColumn : endRow) > getPureColumns() - 1 || (endDepth > getPureDepth() - 1)) {
            throw new MatrixException("Slice rows: " + startRow + " - " + endRow + " and slice columns: " + startColumn + " - " + endColumn + " and slice depth: " + startDepth + " - " + endDepth + " do not match matrix dimensions: " + getTotalRows() + "x" + getTotalColumns() + "x" + getTotalDepth());
        }
        else updateSliceDimensions(startRow, startColumn, startDepth, (!isTransposed() ? endRow : endColumn), (!isTransposed() ? endColumn : endRow), endDepth);
    }

    /**
     * Removes slicing of matrix.
     *
     * @throws MatrixException throws exception if matrix cannot be sliced.
     */
    public void unslice() throws MatrixException {
        if (!canBeSliced()) throw new MatrixException("Matrix cannot be sliced.");
        updateSliceDimensions(0, 0, 0, getPureRows() - 1, getPureColumns() - 1, getPureDepth() - 1);
    }

    /**
     * Updates slice dimensions.
     *
     * @param startRow slice start row
     * @param startColumn slice start columns
     * @param startDepth slice start depth
     * @param endRow slice end row
     * @param endColumn slide end column
     * @param endDepth slide end depth
     */
    private void updateSliceDimensions(int startRow, int startColumn, int startDepth, int endRow, int endColumn, int endDepth) {
        sliceStartRow = startRow;
        sliceStartColumn = startColumn;
        sliceStartDepth = startDepth;
        sliceRows = endRow - sliceStartRow + 1;
        sliceColumns = endColumn - sliceStartColumn + 1;
        sliceDepth = endDepth - sliceStartDepth + 1;
        sliceSize = sliceRows * sliceColumns;
    }

    /**
     * Returns size (rows * columns) of matrix
     *
     * @return size of matrix.
     */
    public int size() {
        return canBeSliced() ? sliceSize : getPureRows() * getPureColumns();
    }

    /**
     * Returns number of rows in matrix.
     *
     * @return number of rows in matrix.
     */
    public int getRows() {
        return !isTransposed() ? (canBeSliced() ? getSliceRows() : getPureRows()) : (canBeSliced() ? getSliceColumns() : getPureColumns());
    }

    /**
     * Returns number of columns in matrix.
     *
     * @return number of columns in matrix.
     */
    public int getColumns() {
        return !isTransposed() ? (canBeSliced() ? getSliceColumns() : getPureColumns()) : (canBeSliced() ? getSliceRows() : getPureRows());
    }

    /**
     * Returns depth of matrix.
     *
     * @return depth of matrix.
     */
    public int getDepth() {
        return (canBeSliced() ? getSliceDepth() : getPureDepth());
    }

    /**
     * Sets name for matrix.
     *
     * @param name matrix name.
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Returns name of matrix.
     *
     * @return name of matrix.
     */
    public String getName() {
        return name;
    }

    /**
     * Function used to reinitialize matrix and it's mask.
     *
     */
    public void reset() {
        resetMatrix();
        if (getMask() != null) getMask().reset();
    }

    /**
     * Resets matrix.
     *
     */
    protected abstract void resetMatrix();

    /**
     * Creates new matrix with object reference to the matrix data of this matrix.
     *
     * @return newly created reference matrix.
     * @throws MatrixException throws exception if mask operation fails or cloning of matrix fails.
     */
    public Matrix reference() throws MatrixException {
        Matrix newMatrix;
        // Make shallow copy of matrix leaving references internal objects which are shared.
        try {
            newMatrix = (Matrix)super.clone();
            if (getMask() != null) newMatrix.setMask(getMask().reference());
        } catch (CloneNotSupportedException exception) {
            throw new MatrixException("Cloning of matrix failed.");
        }
        return newMatrix;
    }

    /**
     * Checks if this matrix and other matrix are equal in dimensions (rows x columns).
     *
     * @param other other matrix to be compared against.
     * @return true if matrices are of same size otherwise false.
     */
    public boolean hasEqualSize(Matrix other) {
        return other.getRows() == getRows() && other.getColumns() == getColumns() && other.getDepth() == getDepth();
    }

    /**
     * Sets procedure factory for matrix.
     *
     * @param procedureFactory new procedure factory.
     */
    public void setProcedureFactory(ProcedureFactory procedureFactory) {
        this.procedureFactory = procedureFactory;
    }

    /**
     * Returns current procedure factory of matrix.
     *
     * @return current procedure factory.
     */
    public ProcedureFactory getProcedureFactory() {
        return procedureFactory;
    }

    /**
     * Removes procedure factory.
     *
     */
    public void removeProcedureFactory() {
        this.procedureFactory = null;
    }

    /**
     * Returns true if matrix has procedure factory otherwise false.
     *
     * @return true if matrix has procedure factory otherwise false.
     */
    public boolean hasProcedureFactory() {
        return procedureFactory != null;
    }

    /**
     * Returns new matrix of same dimensions.
     *
     * @return new matrix of same dimensions.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public Matrix getNewMatrix() throws MatrixException {
        return getNewMatrix(false);
    }

    /**
     * Returns new matrix of same dimensions optionally as transposed.
     *
     * @param asTransposed if true returns new matrix as transposed otherwise with unchanged dimensions.
     * @return new matrix of same dimensions.
     * @throws MatrixException throws exception if new mask dimensions or mask type are not matching with this mask.
     */
    public Matrix getNewMatrix(boolean asTransposed) throws MatrixException {
        return isScalar() ? getNewMatrix(0) : !asTransposed ? getNewMatrix(getRows(), getColumns(), getDepth()) :  getNewMatrix(getColumns(), getRows(), getDepth());
    }

    /**
     * Checks if data of other matrix is equal to data of this matrix
     *
     * @param other matrix to be compared.
     * @return true is data of this and other matrix are equal otherwise false.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public boolean equals(Matrix other) throws MatrixException {
        return new IsEqualMatrixOperation(getRows(), getColumns(), getDepth()).apply(this, other);
    }

    /**
     * Makes current matrix data equal to other matrix data.
     *
     * @param other other matrix to be copied as data of this matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void setEqualTo(Matrix other) throws MatrixException {
        new EqualMatrixOperation(getRows(), getColumns(), getDepth()).apply(other, this);
    }

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param unaryFunction unary function.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return result matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix apply(UnaryFunction unaryFunction, boolean inplace) throws MatrixException {
        if (!hasProcedureFactory()) return applyFunction(unaryFunction, inplace);
        else {
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyFunction(unaryFunction, inplace);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createUnaryFunctionExpression(expressionLock, this, result, unaryFunction);
            return result;
        }
    }

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param unaryFunction unaryFunction to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix apply(UnaryFunction unaryFunction) throws MatrixException {
        return apply(unaryFunction, false);
    }

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param unaryFunctionType unaryFunction type to be applied.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix apply(UnaryFunctionType unaryFunctionType, boolean inplace) throws MatrixException, DynamicParamException {
        return apply(new UnaryFunction(unaryFunctionType), inplace);
    }

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param unaryFunctionType unaryFunction type to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix apply(UnaryFunctionType unaryFunctionType) throws MatrixException, DynamicParamException {
        return apply(unaryFunctionType, false);
    }

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param unaryFunction unary function.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return result matrix.
     * @throws MatrixException not thrown in any situation.
     */
    private Matrix applyFunction(UnaryFunction unaryFunction, boolean inplace) throws MatrixException {
        return new UnaryMatrixOperation(getRows(), getColumns(), getDepth(), unaryFunction).applyFunction(this, inplace);
    }

    /**
     * Applies two variable operation to this matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other other matrix
     * @param binaryFunction binaryFunction to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyBi(Matrix other, BinaryFunction binaryFunction) throws MatrixException {
        return applyBi(other, binaryFunction, false);
    }

    /**
     * Applies two variable operation to this matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other          other matrix
     * @param binaryFunction binaryFunction to be applied.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return matrix which stores operation result.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyBi(Matrix other, BinaryFunction binaryFunction, boolean inplace) throws MatrixException {
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) return applyBiFunction(other, binaryFunction, inplace);
        else {
            ProcedureFactory.synchronize(this, other);
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyBiFunction(other, binaryFunction, inplace);
            ProcedureFactory.synchronize(this, other, result);
            getProcedureFactory().createBinaryFunctionExpression(expressionLock, this, other, result, binaryFunction);
            return result;
        }
    }

    /**
     * Applies two variable operation to this matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other              other matrix
     * @param binaryFunctionType binaryFunction type to be applied.
     * @throws MatrixException       throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix applyBi(Matrix other, BinaryFunctionType binaryFunctionType) throws MatrixException, DynamicParamException {
        return applyBi(other, new BinaryFunction(binaryFunctionType));
    }

    /**
     * Applies two variable operation to this matrix and other matrix and stores operation result into result matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix or
     * multiplying current matrix with other matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param binaryFunction binary function.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     */
    private Matrix applyBiFunction(Matrix other, BinaryFunction binaryFunction, boolean inplace) throws MatrixException {
        // Checks if there is need to broadcast or un-broadcast due to scalar matrix.
        int rows = !isScalar() ? getRows() : other.getRows();
        int columns = !isScalar() ? getColumns() : other.getColumns();
        return new BinaryMatrixOperation(rows, columns, getDepth(), binaryFunction).applyFunction(this, other, inplace);
    }

    /**
     * Adds other matrix to this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    private Matrix add(Matrix other, boolean inplace) throws MatrixException {
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) return applyBi (other, new BinaryFunction((MatrixBinaryOperation & Serializable) Double::sum), inplace);
        else {
            ProcedureFactory.synchronize(this, other);
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyBi (other, new BinaryFunction((MatrixBinaryOperation & Serializable) Double::sum), inplace);
            ProcedureFactory.synchronize(this, other, result);
            getProcedureFactory().createAddExpression(expressionLock, this, other, result);
            return result;
        }
    }

    /**
     * Adds first matrices element wise with second matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param firstMatrices first matrices for operation.
     * @param secondMatrix second matrix for operation.
     * @return matrices which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public static TreeMap<Integer, Matrix> add(TreeMap<Integer, Matrix> firstMatrices, Matrix secondMatrix) throws MatrixException {
        Matrix firstMatrix = firstMatrices.get(firstMatrices.firstKey());
        if (!firstMatrix.hasProcedureFactory() && !secondMatrix.hasProcedureFactory()) return AbstractMatrix.applyAdd (firstMatrices, secondMatrix);
        else {
            ProcedureFactory.synchronize(firstMatrix, secondMatrix);
            int expressionLock = firstMatrix.getProcedureFactory().startExpression();
            TreeMap<Integer, Matrix> resultMatrices = AbstractMatrix.applyAdd (firstMatrices, secondMatrix);
            Matrix resultMatrix = resultMatrices.get(resultMatrices.firstKey());
            ProcedureFactory.synchronize(firstMatrix, secondMatrix, resultMatrix);
            firstMatrix.getProcedureFactory().createAddExpression(expressionLock, firstMatrix, secondMatrix, resultMatrix);
            return resultMatrices;
        }
    }

    /**
     * Adds first matrices element wise with second matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param firstMatrices first matrices for operation.
     * @param secondMatrix second matrix for operation.
     * @return result matrices
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public static TreeMap<Integer, Matrix> applyAdd(TreeMap<Integer, Matrix> firstMatrices, Matrix secondMatrix) throws MatrixException {
        TreeMap<Integer, Matrix> resultMatrices = new TreeMap<>();
        for (Map.Entry<Integer, Matrix> entry : firstMatrices.entrySet()) {
            resultMatrices.put(entry.getKey(), entry.getValue().add(secondMatrix));
        }
        return resultMatrices;
    }

    /**
     * Adds other matrix to this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public Matrix add(Matrix other) throws MatrixException {
        return add(other, false);
    }

    /**
     * Adds constant number to this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant contains constant value to be added.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix add(double constant) throws MatrixException {
        return add(getNewMatrix(constant));
    }

    /**
     * Adds this matrix by other matrix.
     *
     * @param other other matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void addBy(Matrix other) throws MatrixException {
        add(other, true);
    }

    /**
     * Adds this matrix by constant.
     *
     * @param constant constant.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void addBy(double constant) throws MatrixException {
        add(getNewMatrix(constant), true);
    }

    /**
     * Increment value of specific row, column and depth.
     *
     * @param row row of value to be added.
     * @param column column of value to be added.
     * @param depth depth of value to be added.
     * @param value to be added.
     */
    public void addByValue(int row, int column, int depth, double value) {
        setValue(row, column, depth, getValue(row, column, depth) + value);
    }

    /**
     * Subtracts other matrix from this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    private Matrix subtract(Matrix other, boolean inplace) throws MatrixException {
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) return applyBi (other, new BinaryFunction((MatrixBinaryOperation & Serializable) (value1, value2) -> value1 - value2), inplace);
        else {
            ProcedureFactory.synchronize(this, other);
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyBi (other, new BinaryFunction((MatrixBinaryOperation & Serializable) (value1, value2) -> value1 - value2), inplace);
            ProcedureFactory.synchronize(this, other, result);
            getProcedureFactory().createSubtractExpression(expressionLock, this, other, result);
            return result;
        }
    }

    /**
     * Subtracts first matrices element wise with second matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param firstMatrices first matrices for operation.
     * @param secondMatrix second matrix for operation.
     * @return matrices which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public static TreeMap<Integer, Matrix> subtract(TreeMap<Integer, Matrix> firstMatrices, Matrix secondMatrix) throws MatrixException {
        Matrix firstMatrix = firstMatrices.get(firstMatrices.firstKey());
        if (!firstMatrix.hasProcedureFactory() && !secondMatrix.hasProcedureFactory()) return AbstractMatrix.applySubtract (firstMatrices, secondMatrix);
        else {
            ProcedureFactory.synchronize(firstMatrix, secondMatrix);
            int expressionLock = firstMatrix.getProcedureFactory().startExpression();
            TreeMap<Integer, Matrix> resultMatrices = AbstractMatrix.applySubtract (firstMatrices, secondMatrix);
            Matrix resultMatrix = resultMatrices.get(resultMatrices.firstKey());
            ProcedureFactory.synchronize(firstMatrix, secondMatrix, resultMatrix);
            firstMatrix.getProcedureFactory().createSubtractExpression(expressionLock, firstMatrix, secondMatrix, resultMatrix);
            return resultMatrices;
        }
    }

    /**
     * Subtracts first matrices element wise with second matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param firstMatrices first matrices for operation.
     * @param secondMatrix second matrix for operation.
     * @return result matrices
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    private static TreeMap<Integer, Matrix> applySubtract(TreeMap<Integer, Matrix> firstMatrices, Matrix secondMatrix) throws MatrixException {
        TreeMap<Integer, Matrix> resultMatrices = new TreeMap<>();
        for (Map.Entry<Integer, Matrix> entry : firstMatrices.entrySet()) {
            resultMatrices.put(entry.getKey(), entry.getValue().subtract(secondMatrix));
        }
        return resultMatrices;
    }

    /**
     * Subtracts other matrix from this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public Matrix subtract(Matrix other) throws MatrixException {
        return subtract(other, false);
    }

    /**
     * Subtracts constant number from this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant contains constant value to be subtracted.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix subtract(double constant) throws MatrixException {
        return subtract(getNewMatrix(constant));
    }

    /**
     * Subtracts this matrix by other matrix.
     *
     * @param other other matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void subtractBy(Matrix other) throws MatrixException {
        subtract(other, true);
    }

    /**
     * Subtracts this matrix by constant.
     *
     * @param constant constant.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void subtractBy(double constant) throws MatrixException {
        subtract(getNewMatrix(constant), true);
    }

    /**
     * Decrease value of specific row, column and depth.
     *
     * @param row row of value to be decreased.
     * @param column column of value to be decreased.
     * @param depth depth of value to be decreaeed.
     * @param value to be decreased.
     */
    public void subtractByValue(int row, int column, int depth, double value) {
        setValue(row, column, depth, getValue(row, column, depth) - value);
    }

    /**
     * Multiplies other matrix element wise with this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    private Matrix multiply(Matrix other, boolean inplace) throws MatrixException {
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) return applyBi (other, new BinaryFunction((MatrixBinaryOperation & Serializable) (value1, value2) -> value1 * value2), inplace);
        else {
            ProcedureFactory.synchronize(this, other);
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyBi (other, new BinaryFunction((MatrixBinaryOperation & Serializable) (value1, value2) -> value1 * value2), inplace);
            ProcedureFactory.synchronize(this, other, result);
            getProcedureFactory().createMultiplyExpression(expressionLock, this, other, result);
            return result;
        }
    }

    /**
     * Multiplies first matrices element wise with second matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param firstMatrices first matrices for operation.
     * @param secondMatrix second matrix for operation.
     * @return matrices which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public static TreeMap<Integer, Matrix> multiply(TreeMap<Integer, Matrix> firstMatrices, Matrix secondMatrix) throws MatrixException {
        Matrix firstMatrix = firstMatrices.get(firstMatrices.firstKey());
        if (!firstMatrix.hasProcedureFactory() && !secondMatrix.hasProcedureFactory()) return AbstractMatrix.applyMultiply (firstMatrices, secondMatrix);
        else {
            ProcedureFactory.synchronize(firstMatrix, secondMatrix);
            int expressionLock = firstMatrix.getProcedureFactory().startExpression();
            TreeMap<Integer, Matrix> resultMatrices = AbstractMatrix.applyMultiply (firstMatrices, secondMatrix);
            Matrix resultMatrix = resultMatrices.get(resultMatrices.firstKey());
            ProcedureFactory.synchronize(firstMatrix, secondMatrix, resultMatrix);
            firstMatrix.getProcedureFactory().createMultiplyExpression(expressionLock, firstMatrix, secondMatrix, resultMatrix);
            return resultMatrices;
        }
    }

    /**
     * Multiplies first matrices element wise with second matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param firstMatrices first matrices for operation.
     * @param secondMatrix second matrix for operation.
     * @return result matrices
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    private static TreeMap<Integer, Matrix> applyMultiply(TreeMap<Integer, Matrix> firstMatrices, Matrix secondMatrix) throws MatrixException {
        TreeMap<Integer, Matrix> resultMatrices = new TreeMap<>();
        for (Map.Entry<Integer, Matrix> entry : firstMatrices.entrySet()) {
            resultMatrices.put(entry.getKey(), entry.getValue().multiply(secondMatrix));
        }
        return resultMatrices;
    }

    /**
     * Multiplies other matrix element wise with this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public Matrix multiply(Matrix other) throws MatrixException {
        return multiply(other, false);
    }

    /**
     * Multiplies constant number with this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant contains constant value to be multiplied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix multiply(double constant) throws MatrixException {
        return multiply(getNewMatrix(constant));
    }

    /**
     * Multiplies this matrix by other matrix.
     *
     * @param other other matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void multiplyBy(Matrix other) throws MatrixException {
        multiply(other, true);
    }

    /**
     * Multiplies this matrix by constant.
     *
     * @param constant constant.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void multiplyBy(double constant) throws MatrixException {
        multiply(getNewMatrix(constant), true);
    }

    /**
     * Multiply value of specific row, column and depth.
     *
     * @param row row of value to be multiplied.
     * @param column column of value to be multiplied.
     * @param depth depth of value to be multiplied.
     * @param value to be multiplied.
     */
    public void multiplyByValue(int row, int column, int depth, double value) {
        setValue(row, column, depth, getValue(row, column, depth) * value);
    }

    /**
     * Divides this matrix element wise with other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    private Matrix divide(Matrix other, boolean inplace) throws MatrixException {
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) return applyBi (other, new BinaryFunction((MatrixBinaryOperation & Serializable) (value1, value2) -> value1 / value2), inplace);
        else {
            ProcedureFactory.synchronize(this, other);
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyBi (other, new BinaryFunction((MatrixBinaryOperation & Serializable) (value1, value2) -> value1 / value2), inplace);
            ProcedureFactory.synchronize(this, other, result);
            getProcedureFactory().createDivideExpression(expressionLock, this, other, result);
            return result;
        }
    }

    /**
     * Divides matrices element wise with other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param firstMatrices first matrices for operation.
     * @param secondMatrix second matrix for operation.
     * @return matrices which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public static TreeMap<Integer, Matrix> divide(TreeMap<Integer, Matrix> firstMatrices, Matrix secondMatrix) throws MatrixException {
        Matrix firstMatrix = firstMatrices.get(firstMatrices.firstKey());
        if (!firstMatrix.hasProcedureFactory() && !secondMatrix.hasProcedureFactory()) return AbstractMatrix.applyDivide (firstMatrices, secondMatrix);
        else {
            ProcedureFactory.synchronize(firstMatrix, secondMatrix);
            int expressionLock = firstMatrix.getProcedureFactory().startExpression();
            TreeMap<Integer, Matrix> resultMatrices = AbstractMatrix.applyDivide (firstMatrices, secondMatrix);
            Matrix resultMatrix = resultMatrices.get(resultMatrices.firstKey());
            ProcedureFactory.synchronize(firstMatrix, secondMatrix, resultMatrix);
            firstMatrix.getProcedureFactory().createDivideExpression(expressionLock, firstMatrix, secondMatrix, resultMatrix);
            return resultMatrices;
        }
    }

    /**
     * Divides first matrices element wise with second matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param firstMatrices first matrices for operation.
     * @param secondMatrix second matrix for operation.
     * @return result matrices
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    private static TreeMap<Integer, Matrix> applyDivide(TreeMap<Integer, Matrix> firstMatrices, Matrix secondMatrix) throws MatrixException {
        TreeMap<Integer, Matrix> resultMatrices = new TreeMap<>();
        for (Map.Entry<Integer, Matrix> entry : firstMatrices.entrySet()) {
            resultMatrices.put(entry.getKey(), entry.getValue().divide(secondMatrix));
        }
        return resultMatrices;
    }

    /**
     * Divides this matrix element wise with other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public Matrix divide(Matrix other) throws MatrixException {
        return divide(other, false);
    }

    /**
     * Divides this matrix element wise with constant.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant constant used as divider value.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix divide(double constant) throws MatrixException {
        return divide(getNewMatrix(constant));
    }

    /**
     * Divides this matrix by other matrix.
     *
     * @param other other matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void divideBy(Matrix other) throws MatrixException {
        divide(other, true);
    }

    /**
     * Divides this matrix by constant.
     *
     * @param constant constant.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void divideBy(double constant) throws MatrixException {
        divide(getNewMatrix(constant), true);
    }

    /**
     * Divide value of specific row, column and depth.
     *
     * @param row row of value to be divided.
     * @param column column of value to be divided.
     * @param depth depth of value to be divided.
     * @param value to be divided.
     */
    public void divideByValue(int row, int column, int depth, double value) {
        setValue(row, column, depth, getValue(row, column, depth) / value);
    }

    /**
     *
     * Takes matrix dot product of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if columns of this matrix and rows of other matrix are not matching are not matching.
     */
    public Matrix dot(Matrix other) throws MatrixException {
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) return applyDot(other);
        else {
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyDot(other);
            ProcedureFactory.synchronize(this, other, result);
            getProcedureFactory().createDotExpression(expressionLock, this, other, result);
            return result;
        }
    }

    /**
     * Takes matrix dot product of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if columns of this matrix and rows of other matrix are not matching or rows of this and result matrix or columns of result and other matrix are not matching.
     */
    private Matrix applyDot(Matrix other) throws MatrixException {
        return new DotMatrixOperation(getRows(), other.getRows(), other.getColumns(), getDepth()).apply(this, other);
    }

    /**
     * Raises this matrix element wise to the power of value power.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param power power value to which elements are to be raised.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix power(double power) throws MatrixException, DynamicParamException {
        return applyBi (getNewMatrix(power), BinaryFunctionType.POW);
    }

    /**
     * Takes element wise max value of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix max(Matrix other) throws MatrixException, DynamicParamException {
        return applyBi (other, BinaryFunctionType.MAX);
    }

    /**
     * Takes element wise max value of this and other value.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param other value which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix max(double other) throws MatrixException, DynamicParamException {
        return applyBi (getNewMatrix(other), BinaryFunctionType.MAX);
    }

    /**
     * Takes element wise min value of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix min(Matrix other) throws MatrixException, DynamicParamException {
        return applyBi (other, BinaryFunctionType.MIN);
    }

    /**
     * Takes element wise min value of this and other matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param other value which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix min(double other) throws MatrixException, DynamicParamException {
        return applyBi (getNewMatrix(other), BinaryFunctionType.MIN);
    }

    /**
     * Takes element wise signum over multiplication of this and other matrix.<br>
     * Applies first sign operation to each value and then multiplies them.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix sgnmul(Matrix other) throws MatrixException, DynamicParamException {
        return apply(UnaryFunctionType.SGN).multiply(other.apply(UnaryFunctionType.SGN));
    }

    /**
     * Calculates sum or mean.
     *
     * @param matrices matrices.
     * @param asMean if true returns mean otherwise sum.
     * @return result of sum or mean
     * @throws MatrixException throws exception if row or column vectors are incorrectly provided.
     */
    public static Matrix count(TreeMap<Integer, Matrix> matrices, boolean asMean) throws MatrixException {
        Matrix result = null;
        for (Matrix matrix : matrices.values()) {
            if (result == null) result = matrix.getNewMatrix();
            result.addBy(matrix);
        }
        return asMean ? result == null ? null : result.divide(matrices.size()) : result;
    }

    /**
     * Calculates sum of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return sum of matrix.
     */
    public double sum() throws MatrixException {
        return new SumMatrixOperation(getRows(), getColumns(), getDepth(), 0).applySum(this);
    }

    /**
     * Takes element wise cumulative sum of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return sum of matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix sumAsMatrix(int direction) throws MatrixException {
        if (!hasProcedureFactory()) return applySumAsMatrix(direction);
        else {
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applySumAsMatrix(direction);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createSumExpression(expressionLock, this, result, direction);
            return result;
        }
    }

    /**
     * Calculates sum as matrix.
     *
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return sum as matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix applySumAsMatrix(int direction) throws MatrixException {
        return new SumMatrixOperation(getRows(), getColumns(), getDepth(), direction).applySumAsMatrix(this);
    }

    /**
     * Calculates sum.
     *
     * @param matrices matrices.
     * @return resulting sum
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     */
    public static Matrix sum(TreeMap<Integer, Matrix> matrices) throws MatrixException {
        Matrix firstMatrix = matrices.get(matrices.firstKey());
        if (!firstMatrix.hasProcedureFactory()) return applySum(matrices);
        else {
            int expressionLock = firstMatrix.getProcedureFactory().startExpression();
            Matrix result = applySum(matrices);
            ProcedureFactory.synchronize(firstMatrix, result);
            firstMatrix.getProcedureFactory().createSumExpression(expressionLock, firstMatrix, result, true, 0);
            return result;
        }
    }

    /**
     * Calculates sum.
     *
     * @param matrices matrices.
     * @return resulting sum
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     */
    public static Matrix applySum(TreeMap<Integer, Matrix> matrices) throws MatrixException {
        return AbstractMatrix.count(matrices, false);
    }

    /**
     * Calculates mean of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return mean of matrix.
     */
    public double mean() throws MatrixException {
        return new SumMatrixOperation(getRows(), getColumns(), getDepth(), 0).applyMean(this);
    }

    /**
     * Takes mean of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return mean of matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix meanAsMatrix(int direction) throws MatrixException {
        if (!hasProcedureFactory()) return applyMeanAsMatrix(direction);
        else {
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyMeanAsMatrix(direction);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createMeanExpression(expressionLock, this, result, direction);
            return result;
        }
    }

    /**
     * Calculates mean as matrix.
     *
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return mean as matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix applyMeanAsMatrix(int direction) throws MatrixException {
        return new SumMatrixOperation(getRows(), getColumns(), getDepth(), direction).applyMeanAsMatrix(this);
    }

    /**
     * Calculates mean.
     *
     * @param matrices matrices.
     * @return resulting mean
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     */
    public static Matrix mean(TreeMap<Integer, Matrix> matrices) throws MatrixException {
        Matrix firstMatrix = matrices.get(matrices.firstKey());
        if (!firstMatrix.hasProcedureFactory()) return applyMean(matrices);
        else {
            int expressionLock = firstMatrix.getProcedureFactory().startExpression();
            Matrix result = applyMean(matrices);
            ProcedureFactory.synchronize(firstMatrix, result);
            firstMatrix.getProcedureFactory().createMeanExpression(expressionLock, firstMatrix, result, true, 0);
            return result;
        }
    }

    /**
     * Calculates mean.
     *
     * @param matrices matrices.
     * @return resulting mean
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     */
    public static Matrix applyMean(TreeMap<Integer, Matrix> matrices) throws MatrixException {
        return AbstractMatrix.count(matrices, true);
    }

    /**
     * Takes variance of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return variance of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public double variance() throws MatrixException, DynamicParamException {
        return variance(mean());
    }

    /**
     * Calculates variance of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return variance of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public double variance(double mean) throws MatrixException, DynamicParamException {
        return new VarianceMatrixOperation(getRows(), getColumns(), getDepth(), mean).applyVariance(this);
    }

    /**
     * Takes variance of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return variance of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix varianceAsMatrix(int direction) throws MatrixException, DynamicParamException {
        return varianceAsMatrix(null, direction);
    }

    /**
     * Calculates variance as matrix.
     *
     * @param mean mean matrix
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return variance as matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private Matrix applyVarianceAsMatrix(Matrix mean, int direction) throws MatrixException, DynamicParamException {
        return new VarianceMatrixOperation(getRows(), getColumns(), getDepth(), direction).applyVarianceAsMatrix(this, mean != null ? mean : meanAsMatrix(direction));
    }

    /**
     * Takes variance of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean matrix given as input.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return variance of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix varianceAsMatrix(Matrix mean, int direction) throws MatrixException, DynamicParamException {
        if (!hasProcedureFactory()) return applyVarianceAsMatrix(mean, direction);
        else {
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyVarianceAsMatrix(mean, direction);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createVarianceExpression(expressionLock, this, result, direction);
            return result;
        }
    }

    /**
     * Calculates variance.
     *
     * @param matrices matrices.
     * @param mean matrix containing mean values for variance calculation.
     * @return resulting variance
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static Matrix variance(TreeMap<Integer, Matrix> matrices, Matrix mean) throws MatrixException, DynamicParamException {
        Matrix firstMatrix = matrices.get(matrices.firstKey());
        if (!firstMatrix.hasProcedureFactory()) return applyVariance(matrices, mean);
        else {
            int expressionLock = firstMatrix.getProcedureFactory().startExpression();
            Matrix result = applyVariance(matrices, mean);
            if (result != null) ProcedureFactory.synchronize(firstMatrix, result);
            firstMatrix.getProcedureFactory().createVarianceExpression(expressionLock, firstMatrix, result, true, 0);
            return result;
        }
    }

    /**
     * Calculates variance.
     *
     * @param matrices matrices.
     * @param mean matrix containing mean values for variance calculation.
     * @return resulting variance
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private static Matrix applyVariance(TreeMap<Integer, Matrix> matrices, Matrix mean) throws MatrixException, DynamicParamException {
        if (mean == null) throw new MatrixException("Mean matrix is not defined");
        Matrix result = null;
        for (Matrix matrix : matrices.values()) {
            if (result == null) result = matrix.getNewMatrix();
            result.addBy(matrix.subtract(mean).power(2));
        }
        return result == null ? null : result.divide(matrices.size());
    }

    /**
     * Takes standard deviation of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return standard deviation of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public double standardDeviation() throws MatrixException, DynamicParamException {
        return standardDeviation(mean());
    }

    /**
     * Calculates standard deviation of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return standard deviation of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public double standardDeviation(double mean) throws MatrixException, DynamicParamException {
        return new VarianceMatrixOperation(getRows(), getColumns(), getDepth(), mean).applyStandardDeviation(this);
    }

    /**
     * Takes standard deviation of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return standard deviation of matrix.
     * @throws MatrixException       not thrown in any situation.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix standardDeviationAsMatrix(int direction) throws MatrixException, DynamicParamException {
        return standardDeviationAsMatrix(null, direction);
    }

    /**
     * Takes standard deviation of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return standard deviation of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix standardDeviationAsMatrix(Matrix mean, int direction) throws MatrixException, DynamicParamException {
        if (!hasProcedureFactory()) return applyStandardDeviationAsMatrix(mean, direction);
        else {
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyStandardDeviationAsMatrix(mean, direction);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createStandardDeviationExpression(expressionLock, this, result, direction);
            return result;
        }
    }

    /**
     * Calculates standard deviation as matrix.
     *
     * @param mean mean matrix
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return standard deviation as matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private Matrix applyStandardDeviationAsMatrix(Matrix mean, int direction) throws MatrixException, DynamicParamException {
        return new VarianceMatrixOperation(getRows(), getColumns(), getDepth(), direction).applyStandardDeviationAsMatrix(this, mean != null ? mean : meanAsMatrix(direction));
    }

    /**
     * Calculates standard deviation.
     *
     * @param matrices matrices.
     * @param mean matrix containing mean values for standard deviation calculation.
     * @return resulting standard deviation
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static Matrix standardDeviation(TreeMap<Integer, Matrix> matrices, Matrix mean) throws MatrixException, DynamicParamException {
        Matrix firstMatrix = matrices.get(matrices.firstKey());
        if (!firstMatrix.hasProcedureFactory()) return applyStandardDeviation(matrices, mean);
        else {
            int expressionLock = firstMatrix.getProcedureFactory().startExpression();
            Matrix result = applyStandardDeviation(matrices, mean);
            if (result != null) ProcedureFactory.synchronize(firstMatrix, result);
            firstMatrix.getProcedureFactory().createStandardDeviationExpression(expressionLock, firstMatrix, result, true, 0);
            return result;
        }
    }

    /**
     * Calculates standard deviation.
     *
     * @param matrices matrices.
     * @param mean matrix containing mean values for variance calculation.
     * @return resulting standard deviation.
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private static Matrix applyStandardDeviation(TreeMap<Integer, Matrix> matrices, Matrix mean) throws MatrixException, DynamicParamException {
        if (mean == null) throw new MatrixException("Mean matrix is not defined");
        Matrix result = null;
        for (Matrix matrix : matrices.values()) {
            if (result == null) result = matrix.getNewMatrix();
            result.addBy(matrix.subtract(mean).power(2));
        }
        return result == null ? null : result.divide(matrices.size()).multiply(matrices.size()).divide(matrices.size() - 1).apply(UnaryFunctionType.SQRT);
    }

    /**
     * Calculates cumulative p- norm (p is number equal or bigger than 1) of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @param p p value for norm.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return norm of matrix.
     */
    public double norm(int p) throws MatrixException {
        return new NormMatrixOperation(getRows(), getColumns(), getDepth(), p).apply(this);
    }

    /**
     * Takes cumulative p- norm (p is number equal or bigger than 1) of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param p p value for norm.
     * @return norm of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix normAsMatrix(int p) throws MatrixException {
        if (!hasProcedureFactory()) return getNewMatrix(norm(p));
        else {
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = getNewMatrix(norm(p));
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createNormExpression(expressionLock, this, result, p);
            return result;
        }
    }

    /**
     * Normalizes matrix by removing mean and variance.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @param inplace if true matrix is normalized in place otherwise copy of normalized matrix is returned.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return normalized matrix.
     */
    public Matrix normalize(boolean inplace) throws MatrixException, DynamicParamException {
        return new NormalizeMatrixOperation(getRows(), getColumns(), getDepth(), mean(), variance()).apply(this, inplace ? this : getNewMatrix());
    }

    /**
     * Calculates exponential moving average.
     *
     * @param currentExponentialAverage current average value
     * @param momentum degree of weighting decrease for exponential moving average.
     * @return updated average with new average value included.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix exponentialMovingAverage(Matrix currentExponentialAverage, double momentum) throws MatrixException {
        return currentExponentialAverage == null ? this : currentExponentialAverage.multiply(momentum).add(multiply(1 - momentum));
    }

    /**
     * Calculates cumulative moving average CMAn = CMAn-1 + (currentAverage - CMAn-1) / sampleCount
     *
     * @param currentMovingAverage current cumulative moving average
     * @param sampleCount current sample count
     * @return updated cumulative moving average.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix cumulativeMovingAverage(Matrix currentMovingAverage, int sampleCount) throws MatrixException {
        return currentMovingAverage == null ? this : this.subtract(currentMovingAverage).divide(sampleCount).add(currentMovingAverage);
    }

    /**
     * Normalizes (scales) this matrix to new min and max values.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param newMinimum new minimum value.
     * @param newMaximum new maximum value.
     * @throws MatrixException not thrown in any situation.
     */
    public void minMax(double newMinimum, double newMaximum) throws MatrixException {
        minMax(this, newMinimum, newMaximum);
    }

    /**
     * Normalizes (scales) other matrix to new min and max values.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other other matrix to be scaled.
     * @param newMinimum new minimum value.
     * @param newMaximum new maximum value.
     * @return scaled result matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix minMax(Matrix other, double newMinimum, double newMaximum) throws MatrixException {
        double minimum = other.min();
        double maximum = other.max();
        double delta = maximum - minimum != 0 ? maximum - minimum : 1;
        return other.apply(new UnaryFunction(value -> (value - minimum) / delta * (newMaximum - newMinimum) + newMinimum));
    }

    /**
     * Returns minimum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return minimum value of matrix.
     */
    public double min() throws MatrixException {
        return new MinMatrixOperation(getRows(), getColumns(), getDepth()).applyMin(this);
    }

    /**
     * Returns minimum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return minimum value of matrix.
     */
    public Matrix minAsMatrix() throws MatrixException {
        return getNewMatrix(min());
    }

    /**
     * Returns argmin meaning row and column of matrix containing minimum value.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return array containing row, column and depth in this order that points to minimum value of matrix.
     */
    public int[] argmin() throws MatrixException {
        return new MinMatrixOperation(getRows(), getColumns(), getDepth()).applyArgMin(this);
    }

    /**
     * Returns maximum value of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return maximum value of matrix.
     */
    public double max() throws MatrixException {
        return new MaxMatrixOperation(getRows(), getColumns(), getDepth()).applyMax(this);
    }

    /**
     * Returns maximum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return maximum value of matrix.
     */
    public Matrix maxAsMatrix() throws MatrixException {
        return getNewMatrix(max());
    }

    /**
     * Returns argmax meaning row and column of matrix containing maximum value.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return array containing row, column and depth in this order that points to maximum value of matrix.
     */
    public int[] argmax() throws MatrixException {
        return new MaxMatrixOperation(getRows(), getColumns(), getDepth()).applyArgMax(this);
    }

    /**
     * Calculates entropy of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return sum of matrix.
     */
    public double entropy() throws MatrixException {
        return new EntropyMatrixOperation(getRows(), getColumns(), getDepth()).applyEntropy(this);
    }

    /**
     * Implements inverted drop out.<br>
     * Function selectively masks out certain percentage of node governed by parameter probability during training phase.<br>
     * During training phase it also compensates all remaining inputs by dividing by probability.<br>
     *
     * @param probability probability
     * @param monte_carlo if true is monte carlo dropout otherwise normal dropout.
     * @param inplace if true clipping in done in place otherwise not.
     * @return result of drop out.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix dropout(double probability, boolean monte_carlo, boolean inplace) throws MatrixException {
        if (!hasProcedureFactory()) return applyDropout(probability, inplace);
        else {
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyDropout(probability, inplace);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createDropoutExpression(expressionLock, this, result, probability, monte_carlo);
            return result;
        }
    }

    /**
     * Implements inverted drop out.<br>
     * Function selectively masks out certain percentage of node governed by parameter probability during training phase.<br>
     * During training phase it also compensates all remaining inputs by dividing by probability.<br>
     *
     * @param probability probability
     * @param inplace if true clipping in done in place otherwise not.
     * @return result of drop out.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix applyDropout(double probability, boolean inplace) throws MatrixException {
        return new DropoutMatrixOperation(getRows(), getColumns(), getDepth(), probability).apply(this, inplace);
    }

    /**
     * Clips gradient matrix against threshold.
     *
     * @param threshold threshold.
     * @return clipped gradient matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix gradientClip(double threshold) throws MatrixException {
        Matrix result = this.copy();
        if (hasProcedureFactory()) {
            int expressionLock = getProcedureFactory().startExpression();
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createGradientClippingExpression(expressionLock, this, result, threshold);
        }
        return result;
    }

    /**
     * Implements matrix noising.
     *
     * @param noise noise
     * @param inplace if true clipping in done in place otherwise not.
     * @return result of drop out.
     */
    public Matrix noise(double noise, boolean inplace) throws MatrixException {
        return apply(new UnaryFunction(value -> value + noise * (1 - 2 * random.nextDouble())), inplace);
    }

    /**
     * Returns softmax of this matrix.
     *
     * @return softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix softmax() throws MatrixException, DynamicParamException {
        return softmax(1);
    }

    /**
     * Returns softmax of matrix.
     *
     * @param softmaxTau tau value for Softmax.
     * @return softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix softmax(double softmaxTau) throws MatrixException, DynamicParamException {
        return apply(new UnaryFunction(UnaryFunctionType.SOFTMAX, "asGumbelSoftmax = true, tau = " + softmaxTau));
    }

    /**
     * Returns Gumbel softmax of this matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @return Gumbel softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix gumbelSoftmax() throws MatrixException, DynamicParamException {
        return gumbelSoftmax(1);
    }

    /**
     * Returns Gumbel softmax of matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @param softmaxTau tau value for Softmax.
     * @return Gumbel softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix gumbelSoftmax(double softmaxTau) throws MatrixException, DynamicParamException {
        return apply(new UnaryFunction(UnaryFunctionType.SOFTMAX, "asGumbelSoftmax = true, tau = " + softmaxTau));
    }

    /**
     * Transposes matrix.
     *
     * @return transposed matrix.
     * @throws MatrixException throws exception if cloning of mask fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix transpose() throws MatrixException, DynamicParamException {
        if (!hasProcedureFactory()) return applyTranspose();
        else {
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyTranspose();
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createUnaryFunctionExpression(expressionLock, this, result, new UnaryFunction(UnaryFunctionType.TRANSPOSE));
            return result;
        }
    }

    /**
     * Applies matrix transpose.
     *
     * @return transposed matrix.
     * @throws MatrixException throws exception if cloning of mask fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected abstract Matrix applyTranspose() throws MatrixException, DynamicParamException;

    /**
     * Checks if matrix is transposed.
     *
     * @return true is matrix is transposed otherwise false.
     */
    public boolean isTransposed() {
        return isTransposed;
    }

    /**
     * Classifies matrix assuming multi-label classification.
     *
     * @return classified matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix classify() throws MatrixException {
        return new ClassifyMatrixOperation(getRows(), getColumns(), getDepth()).apply(this, getNewMatrix());
    }

    /**
     * Classifies matrix assuming multi-label classification.
     *
     * @return classified matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix classify(double multiLabelThreshold) throws MatrixException {
        return new ClassifyMatrixOperation(getRows(), getColumns(), getDepth(), multiLabelThreshold).apply(this, getNewMatrix());
    }

    /**
     * Splits matrix at defined position. If splitVertical is true splits vertically otherwise horizontally.
     *
     * @param splitAt split at position
     * @param splitVertically if true splits vertically otherwise horizontally.
     * @return split matrix as JMatrix.
     * @throws MatrixException throws matrix exception if splitting fails.
     *
     */
    public Matrix split(int splitAt, boolean splitVertically) throws MatrixException {
        Matrix result = new SplitMatrixOperation(getRows(), getColumns(), getDepth()).apply(this, splitAt, splitVertically);
        if (hasProcedureFactory()) ProcedureFactory.synchronize(this, result);
        return result;
    }

    /**
     * Joins two matrices either vertically or horizontally.
     *
     * @param other other matrix
     * @param joinedVertically if true joined vertically otherwise horizontally
     * @return joined matrix
     * @throws MatrixException throws matrix exception if joining fails.
     */
    public Matrix join(Matrix other, boolean joinedVertically) throws MatrixException {
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) return applyJoin(other, joinedVertically);
        else {
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyJoin(other, joinedVertically);
            ProcedureFactory.synchronize(this, other, result);
            getProcedureFactory().createJoinExpression(expressionLock, this, other, result, joinedVertically);
            return result;
        }
    }

    /**
     * Joins two matrices either vertically or horizontally.
     *
     * @param other other matrix
     * @param joinedVertically if true joined vertically otherwise horizontally
     * @return result matrix
     * @throws MatrixException throws matrix exception if joining fails.
     */
    private Matrix applyJoin(Matrix other, boolean joinedVertically) throws MatrixException {
        return new JoinMatrixOperation(getRows() + other.getRows(), getColumns() + other.getColumns(), getDepth(), joinedVertically).apply(this, other);
    }

    /**
     * Joins matrices.
     *
     * @param matrices matrices.
     * @param joinedVertically if true MMatrices are joint vertically otherwise horizontally.
     * @return joint matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public static Matrix join(Matrix[] matrices, boolean joinedVertically) throws MatrixException {
        ArrayList<Matrix> subMatrices = new ArrayList<>();
        Collections.addAll(subMatrices, matrices);
        return subMatrices.isEmpty() ? null : new JMatrix(subMatrices, joinedVertically);
    }

    /**
     * Unjoins matrix at specific row and column.
     *
     * @param unjoinAtRow unjoins at row.
     * @return result matrix.
     * @throws MatrixException throws matrix exception if unjoining fails.
     */
    public Matrix unjoin(int unjoinAtRow) throws MatrixException {
        return unjoin(unjoinAtRow, 0, 0, 1, 1, 1);
    }

    /**
     * Unjoins matrix at specific row and column.
     *
     * @param unjoinAtRow unjoins at row.
     * @param unjoinAtColumn unjoins at column.
     * @param unjoinAtDepth unjoins at depth.
     * @param unjoinRows unjoins specific number of rows.
     * @param unjoinColumns unjoins specific number of column.
     * @param unjoinDepth unjoins specific amount of depth.
     * @return result matrix.
     * @throws MatrixException throws matrix exception if unjoining fails.
     */
    public Matrix unjoin(int unjoinAtRow, int unjoinAtColumn, int unjoinAtDepth, int unjoinRows, int unjoinColumns, int unjoinDepth) throws MatrixException {
        if (!hasProcedureFactory()) return applyUnjoin(unjoinAtRow, unjoinAtColumn, unjoinAtDepth, unjoinRows, unjoinColumns, unjoinDepth);
        else {
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyUnjoin(unjoinAtRow, unjoinAtColumn, unjoinAtDepth, unjoinRows, unjoinColumns, unjoinDepth);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createUnjoinExpression(expressionLock, this, result, unjoinAtRow, unjoinAtColumn, unjoinAtDepth);
            return result;
        }
    }

    /**
     * Unjoins matrix into resulting unjoined matrix and potentially unjoined matrices.
     *
     * @param unjoinAtRow unjoins at row.
     * @param unjoinAtColumn unjoins at column.
     * @param unjoinAtDepth unjoins at depth.
     * @param unjoinRows unjoins specific number of rows.
     * @param unjoinColumns unjoins specific number of column.
     * @param unjoinDepth unjoins specific amount of depth.
     * @return result matrix.
     * @throws MatrixException throws matrix exception if unjoining fails.
     */
    private Matrix applyUnjoin(int unjoinAtRow, int unjoinAtColumn, int unjoinAtDepth, int unjoinRows, int unjoinColumns, int unjoinDepth) throws MatrixException {
        return new UnjoinMatrixOperation(unjoinRows, unjoinColumns, unjoinDepth, unjoinAtRow, unjoinAtColumn, unjoinAtDepth).apply(this);
    }

    /**
     * Unjoins matrix.
     *
     * @param matrix matrix.
     * @return unjoined matrices.
     */
    public static Matrix[] unjoin(Matrix matrix) {
        ArrayList<Matrix> subMatrices = matrix.getSubMatrices();
        int subMatricesSize = subMatrices.size();
        Matrix[] unjoinedMatrices = new Matrix[subMatricesSize];
        for (int subMatrixIndex = 0; subMatrixIndex < subMatricesSize; subMatrixIndex++) unjoinedMatrices[subMatrixIndex] = subMatrices.get(subMatrixIndex);
        return unjoinedMatrices;
    }

    /**
     * Flattens matrix into one dimensional column vector (matrix)
     *
     * @return flattened matrix
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix flatten() throws MatrixException {
        if (!hasProcedureFactory()) return applyFlatten();
        else {
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyFlatten();
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createFlattenExpression(expressionLock, this, result);
            return result;
        }
    }

    /**
     * Flattens matrix into one dimensional column vector (matrix)
     *
     * @return result matrix
     * @throws MatrixException throws matrix exception if joining fails.
     */
    private Matrix applyFlatten() throws MatrixException {
        return new FlattenMatrixOperation(getRows(), getColumns(), getDepth()).apply(this);
    }

    /**
     * Returns unflattened matrix i.e. samples that have been unflattened from single column vector.
     *
     * @param rows rows of unflattened matrix.
     * @param columns columns of unflattened matrix.
     * @param depth depth of unflattened matrix.
     * @return unflattened matrix.
     * @throws MatrixException throws matrix exception if joining fails.
     */
    public Matrix unflatten(int rows, int columns, int depth) throws MatrixException {
        return new UnflattenMatrixOperation(rows, columns, depth).apply(this);
    }

    /**
     * Encodes value to bit column vector.
     *
     * @param value value
     * @param numberOfBits number of bits.
     * @return bit column vector.
     * @throws MatrixException throws exception if binary code size is exceeding number of bits.
     */
    public static Matrix encodeValueToBitColumnVector(int value, int numberOfBits) throws MatrixException {
        if (value < 0 || numberOfBits <= 0) throw new MatrixException("Invalid input values");
        if (value >= (1 << numberOfBits)) throw new MatrixException("Binary code size exceeds number of available bits");

        Matrix result = new SMatrix(numberOfBits, 1, 1);
        for (int row = 0; row < numberOfBits; row++) {
            result.setValue(row, 0, 0, (value & (1 << row)) >> row);
        }
        return result;
    }

    /**
     * Decodes value from bit column vector.
     *
     * @param encodedMatrix encoded matrix.
     * @return encoded value.
     */
    public static int decodeValueFromBitColumnVector(Matrix encodedMatrix) {
        int maxBits = encodedMatrix.getRows();
        int decodedValue = 0;

        for (int bitIndex = 0; bitIndex < maxBits; bitIndex++) {
            double bitValue = encodedMatrix.getValue(bitIndex, 0, 0);
            if (bitValue == 1) {
                decodedValue += Math.pow(2, maxBits - 1 - bitIndex);
            }
        }

        return decodedValue;
    }

    /**
     * Encodes bit column vector value
     *
     * @return value
     * @throws MatrixException throws exception if matrix is not bit column vector.
     */
    public int encodeBitColumnVectorToValue() throws MatrixException {
        if (getColumns() != 1) throw new MatrixException("Matrix must be column vector.");
        int rows = getRows();
        int result = 0;
        for (int row = 0; row < rows; row++) {
            double value = getValue(row, 0, 0);
            if (!(value == 0 || value == 1)) throw new MatrixException("Bit column vector must contains values of 0 or 1.");
            result += value * Math.pow(2, (rows - 1) - row);
        }
        return result;
    }

    /**
     * Returns number of bits needed to represent value.
     *
     * @param value value.
     * @return number of bits needed to represent value.
     */
    public static int numberOfBits(int value) {
        return (int)Math.floor((Math.log10(value) / Math.log10(2) + 1));
    }

    /**
     * Returns binomial distribution.
     * Reference: <a href="https://peterchng.com/blog/2020/10/23/building-binomial-and-multinomial-samplers-in-java/">...</a>
     *
     * @param probability probability.
     * @return number of successful trials.
     */
    public static int getBinomial(double probability) {
        return getBinomial(1, probability);
    }

    /**
     * Returns binomial distribution.
     * Reference: <a href="https://peterchng.com/blog/2020/10/23/building-binomial-and-multinomial-samplers-in-java/">...</a>
     *
     * @param numberOfTrials number of trials.
     * @param probability probability.
     * @return number of successful trials.
     */
    public static int getBinomial(int numberOfTrials, double probability) {
        if (numberOfTrials < 1 || probability < 0) return 0;
        if (probability > 1) return numberOfTrials;

        int numberOfSuccessfulTrials = 0;
        for (int trial = 0; trial < numberOfTrials; trial++) {
            if (Math.random() < probability) numberOfSuccessfulTrials++;
        }
        return numberOfSuccessfulTrials;
    }

    /**
     * Returns multinomial distribution. Assumes single trial.
     *
     * @return multinomial distribution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getMultinomial() throws MatrixException {
        return getMultinomial(1);
    }

    /**
     * Returns multinomial distribution.
     * Reference: <a href="https://peterchng.com/blog/2020/10/23/building-binomial-and-multinomial-samplers-in-java/">...</a>
     *
     * @param numberOfTrials number of trials.
     * @return multinomial distribution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getMultinomial(int numberOfTrials) throws MatrixException {
        if (numberOfTrials < 1) throw new MatrixException("Number of trials cannot be less than 1.");

        Matrix result = getNewMatrix();
        final int rows = getRows();
        final int columns = getColumns();
        final int totalDepth = getDepth();
        double sumLeft = 1.0;
        int trialsLeft = numberOfTrials;
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                for (int depth = 0; depth < totalDepth; depth++) {
                    double probability = getValue(row, column, depth);
                    double binomial = getBinomial(trialsLeft, probability / sumLeft);
                    result.setValue(row, column, depth, binomial);
                    sumLeft -= probability;
                    trialsLeft -= binomial;
                    if (sumLeft <= 0 || trialsLeft <= 0) break;
                }
            }
        }

        return result;
    }

    /**
     * Samples random variable from gamma distribution.<br>
     * Reference: <a href="https://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/">...</a>
     *
     * @param shape shape (alpha) parameter
     * @param scale scale (beta) parameter
     * @param random random function
     * @return random variable from gamma distribution
     */
    public static double sampleGamma(double shape, double scale, Random random) {
        if (shape > 1) {
            double d = shape - 1 / (double)3;
            double c = 1 / Math.sqrt(9 * d);
            while (true) {
                double gaussian = random.nextGaussian();
                if (gaussian > - 1 / c) {
                    double uniform = random.nextDouble();
                    double V = Math.pow(1 + c * gaussian, 3);
                    if (Math.log(uniform) < 0.5 * Math.pow(gaussian, 2) + d - d * V + d * Math.log(V)) return d * V / scale;
                }
            }
        }
        else return sampleGamma(shape + 1, scale, random) * Math.pow(random.nextDouble(), 1 / shape);
    }

    /**
     * Samples entry from matrix by taking random choice.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return array containing row, column and depth in this order that points to maximum value of matrix.
     */
    public int[] sample() throws MatrixException {
        return new SampleMatrixOperation(getRows(), getColumns(), getDepth()).sample(this);
    }

    /**
     * Sets stride size for convolution and pooling operations.
     *
     * @param stride stride size.
     */
    public void setStride(int stride) {
        this.stride = stride;
    }

    /**
     * Returns stride size for convolution and pooling operations.
     *
     * @return stride size.
     */
    public int getStride() {
        return stride;
    }

    /**
     * Sets dilation step size for convolution operations.
     *
     * @param dilation dilation step size.
     */
    public void setDilation(int dilation) {
        this.dilation = dilation;
    }

    /**
     * Returns dilation step size for convolution operations.
     *
     * @return dilation step size.
     */
    public int getDilation() {
        return dilation;
    }

    /**
     * Sets filter row size for convolution and pooling operations.
     *
     * @param filterRowSize filter row size.
     */
    public void setFilterRowSize(int filterRowSize) {
        this.filterRowSize = filterRowSize;
    }

    /**
     * Sets filter column size for convolution and pooling operations.
     *
     * @param filterColumnSize filter column size.
     */
    public void setFilterColumnSize(int filterColumnSize) {
        this.filterColumnSize = filterColumnSize;
    }

    /**
     * Sets filter depth.
     *
     * @param filterDepth filter depth.
     */
    public void setFilterDepth(int filterDepth) {
        this.filterDepth = filterDepth;
    }

    /**
     * Returns filter row size for convolution and pooling operations.
     *
     * @return filter row size for convolution and pooling operations.
     */
    public int getFilterRowSize() {
        return filterRowSize;
    }

    /**
     * Returns filter column size for convolution and pooling operations.
     *
     * @return filter column size for convolution and pooling operations.
     */
    public int getFilterColumnSize() {
        return filterColumnSize;
    }

    /**
     * Returns filter depth.
     *
     * @return filter depth.
     */
    public int getFilterDepth() {
        return filterDepth;
    }

    /**
     * Sets if convolution is depth separable.
     *
     * @param isDepthSeparable is true convolution is depth separable.
     */
    public void setIsDepthSeparable(boolean isDepthSeparable) {
        this.isDepthSeparable = isDepthSeparable;
    }

    /**
     * Returns if convolution is depth separable.
     *
     * @return if true convolution is depth separable.
     */
    public boolean getIsDepthSeparable() {
        return isDepthSeparable;
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated result of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix convolve(Matrix filter) throws MatrixException {
        if (!hasProcedureFactory() && !filter.hasProcedureFactory()) return applyConvolve(filter);
        else {
            ProcedureFactory.synchronize(this, filter);
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyConvolve(filter);
            ProcedureFactory.synchronize(this, filter, result);
            getProcedureFactory().createConvolveExpression(expressionLock, this, filter, result, getStride(), getDilation(), getIsDepthSeparable());
            return result;
        }
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix applyConvolve(Matrix filter) throws MatrixException {
        return new ConvolutionMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getFilterDepth(), getDepth(), filter.getRows(), filter.getColumns(), getDilation(), getStride(), getIsDepthSeparable()).apply(this, filter);
    }

    /**
     * Calculates crosscorrelation between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated result of crosscorrelation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix crosscorrelate(Matrix filter) throws MatrixException {
        if (!hasProcedureFactory() && !filter.hasProcedureFactory()) return applyCrosscorrelate(filter);
        else {
            ProcedureFactory.synchronize(this, filter);
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyCrosscorrelate(filter);
            ProcedureFactory.synchronize(this, filter, result);
            getProcedureFactory().createCrosscorrelateExpression(expressionLock, this, filter, result, getStride(), getDilation(), getIsDepthSeparable());
            return result;
        }
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix applyCrosscorrelate(Matrix filter) throws MatrixException {
        return new CrosscorrelationMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getFilterDepth(), getDepth(), filter.getRows(), filter.getColumns(), getDilation(), getStride(), getIsDepthSeparable()).apply(this, filter);
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated value of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix winogradConvolve(Matrix filter) throws MatrixException, DynamicParamException {
        if (!hasProcedureFactory() && !filter.hasProcedureFactory()) return applyWinogradConvolve(filter);
        else {
            ProcedureFactory.synchronize(this, filter);
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyWinogradConvolve(filter);
            ProcedureFactory.synchronize(this, filter, result);
            getProcedureFactory().createWinogradConvolveExpression(expressionLock, this, filter, result, getStride(), getDilation());
            return result;
        }
    }

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private Matrix applyWinogradConvolve(Matrix filter) throws MatrixException, DynamicParamException {
        return new WinogradConvolutionMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getFilterDepth()).apply(this, filter);
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     * @param G G matrix
     * @param GT G transposed matrix
     * @return calculated value of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix winogradConvolve(Matrix filter, Matrix A, Matrix AT, Matrix C, Matrix CT, Matrix G, Matrix GT) throws MatrixException, DynamicParamException {
        if (!hasProcedureFactory() && !filter.hasProcedureFactory()) return applyWinogradConvolve(filter, A, AT, C, CT, G, GT);
        else {
            ProcedureFactory.synchronize(this, filter);
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyWinogradConvolve(filter, A, AT, C, CT, G, GT);
            ProcedureFactory.synchronize(this, filter, result);
            getProcedureFactory().createWinogradConvolveExpression(expressionLock, this, filter, result, getStride(), getDilation());
            return result;
        }
    }

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     * @param G G matrix
     * @param GT G transposed matrix
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix applyWinogradConvolve(Matrix filter, Matrix A, Matrix AT, Matrix C, Matrix CT, Matrix G, Matrix GT) throws MatrixException {
        return new WinogradConvolutionMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getFilterDepth(), A, AT, C, CT, G, GT).apply(this, filter);
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param preprocessedFilter preprocessed filter matrix.
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     * @return calculated value of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix winogradConvolve(Matrix preprocessedFilter, Matrix A, Matrix AT, Matrix C, Matrix CT) throws MatrixException, DynamicParamException {
        Matrix result = getNewMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getDepth());
        if (!hasProcedureFactory() && !preprocessedFilter.hasProcedureFactory()) return applyWinogradConvolve(preprocessedFilter, A, AT, C, CT);
        else {
            ProcedureFactory.synchronize(this, preprocessedFilter, result);
            int expressionLock = getProcedureFactory().startExpression();
            applyWinogradConvolve(preprocessedFilter, A, AT, C, CT);
            getProcedureFactory().createWinogradConvolveExpression(expressionLock, this, preprocessedFilter, result, getStride(), getDilation());
            return result;
        }
    }

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param preprocessedFilter preprocessed filter matrix.
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix applyWinogradConvolve(Matrix preprocessedFilter, Matrix A, Matrix AT, Matrix C, Matrix CT) throws MatrixException {
        return new WinogradConvolutionMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getFilterDepth(), A, AT, C, CT).apply(this, preprocessedFilter);
    }

    /**
     * Calculates max pooling operation for this matrix.
     *
     * @param maxPos maximum positions for each row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix maxPool(HashMap<Integer, Integer> maxPos) throws MatrixException {
        if (!hasProcedureFactory()) return applyMaxPool(maxPos);
        else {
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyMaxPool(maxPos);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createMaxPoolExpression(expressionLock, this, result, getDilation(), getStride(), getFilterRowSize(), getFilterColumnSize());
            return result;
        }
    }

    /**
     * Calculates max pooling operation for matrix and returns max arguments.
     *
     * @param maxPos maximum position for each result row and column value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix applyMaxPool(HashMap<Integer, Integer> maxPos) throws MatrixException {
        return new MaxPoolMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getDepth(), getFilterRowSize(), getFilterColumnSize(), getDilation(), getStride()).apply(this, maxPos);
    }

    /**
     * Calculates random pooling operation for this matrix.
     *
     * @param inputPos input positions for each row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix randomPool(HashMap<Integer, Integer> inputPos) throws MatrixException {
        if (!hasProcedureFactory()) return applyRandomPool(inputPos);
        else {
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyRandomPool(inputPos);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createRandomPoolExpression(expressionLock, this, result, getDilation(), getStride(), getFilterRowSize(), getFilterColumnSize());
            return result;
        }
    }

    /**
     * Calculates random pooling operation for matrix and returns input positions.
     *
     * @param inputPos input position for each result row and column value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix applyRandomPool(HashMap<Integer, Integer> inputPos) throws MatrixException {
        return new RandomPoolMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getDepth(), getFilterRowSize(), getFilterColumnSize(), getDilation(), getStride()).apply(this, inputPos);
    }

    /**
     * Calculates cyclic pooling operation for this matrix.
     *
     * @param inputPos input positions for each row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix cyclicPool(HashMap<Integer, Integer> inputPos) throws MatrixException {
        if (!hasProcedureFactory()) return applyCyclicPool(inputPos);
        else {
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyCyclicPool(inputPos);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createCyclicPoolExpression(expressionLock, this, result, getDilation(), getStride(), getFilterRowSize(), getFilterColumnSize());
            return result;
        }
    }

    /**
     * Calculates cyclic pooling operation for matrix and returns input positions.
     *
     * @param inputPos input position for each result row and column value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix applyCyclicPool(HashMap<Integer, Integer> inputPos) throws MatrixException {
        return new CyclicPoolMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getDepth(), getFilterRowSize(), getFilterColumnSize(), getDilation(), getStride()).apply(this, inputPos);
    }

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix averagePool() throws MatrixException {
        if (!hasProcedureFactory()) return applyAveragePool();
        else {
            int expressionLock = getProcedureFactory().startExpression();
            Matrix result = applyAveragePool();
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createAveragePoolExpression(expressionLock, this, result, getDilation(), getStride(), getFilterRowSize(), getFilterColumnSize());
            return result;
        }
    }

    /**
     * Calculates average pooling operation for matrix.
     *
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix applyAveragePool() throws MatrixException {
        return new AveragePoolMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getDepth(), getFilterRowSize(), getFilterColumnSize(), getDilation(), getStride()).apply(this);
    }

    /**
     * Prints matrix in row and column format.
     *
     */
    public void print() {
        int rows = getRows();
        int columns = getColumns();
        int totalDepth = getDepth();
        for (int depth = 0; depth < totalDepth; depth++) {
            for (int row = 0; row < rows; row++) {
                System.out.print("[");
                for (int column = 0; column < columns; column++) {
                    System.out.print(getValue(row, column, depth));
                    if (column < columns - 1) System.out.print(" ");
                }
                System.out.println("]");
            }
        }
    }

    /**
     * Prints size (rows x columns x depth) of matrix.
     *
     */
    public void printSize() {
        System.out.println("Matrix size: " + getRows() + "x" + getColumns() + "x" + getDepth());
    }

    /**
     * Sets mask to this matrix.
     *
     * @param newMask new mask as input.
     * @throws MatrixException throws exception if new mask dimensions or mask type are not matching with this mask.
     */
    public void setMask(Mask newMask) throws MatrixException {
        if (getRows() != newMask.getRows() || getColumns() != newMask.getColumns() || getDepth() != newMask.getDepth()) throw new MatrixException("Dimensions of new mask are not matching with matrix dimensions.");
        if ((this instanceof DMatrix) && !((newMask instanceof DMask))) throw new MatrixException("New mask is of type DMask which is not matching type of matrix (DMatrix)");
        if ((this instanceof SMatrix) && !((newMask instanceof SMask))) throw new MatrixException("New mask is of type SMask which is not matching type of matrix (SMatrix)");
        mask = newMask;
    }

    /**
     * Sets mask to this matrix.
     *
     */
    public void setMask() {
        if (mask == null) mask = getNewMask();
    }

    /**
     * Removes mask from this matrix.
     *
     */
    public void unsetMask() {
        mask = null;
    }

    /**
     * Returns mask of this matrix.
     *
     * @return mask of this matrix.
     */
    public Mask getMask() {
        return mask;
    }

    /**
     * Checks if mask has been set at specific position.
     *
     * @param row specific row.
     * @param column specific column.
     * @param depth specific depth.
     * @return if true mask exists and is masked at specific position (row + column).
     */
    public boolean hasMaskAt(int row, int column, int depth) {
        return getMask() != null && getMask().getMask(row, column, depth);
    }

    /**
     * Returns new mask for this matrix.<br>
     * Implemented by underlying matrix class.<br>
     *
     * @return mask of this matrix.
     */
    protected abstract Mask getNewMask();

}
