/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix;

import utils.configurable.DynamicParamException;
import utils.matrix.operation.JoinMatrixOperation;
import utils.matrix.operation.UnjoinMatrixOperation;
import utils.procedure.ProcedureFactory;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;

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
     * Size of slice (sliceRows * sliceColumns)
     *
     */
    private int sliceSize;

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
     * Constructor for matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     */
    protected AbstractMatrix(int rows, int columns) {
        this(rows, columns, false);
    }

    /**
     * Constructor for abstract matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param isTransposed if true matrix is transposed and if false not transposed.
     */
    protected AbstractMatrix(int rows, int columns, boolean isTransposed) {
        this.rows = rows;
        this.columns = columns;
        this.isTransposed = isTransposed;
    }

    /**
     * Sets parameters for matrix.
     *
     * @param matrix matrix.
     * @throws MatrixException throws exception if cloning of mask fails.
     */
    protected void setParameters(Matrix matrix) throws MatrixException {
        matrix.setName(name);
        if (getMask() != null) matrix.setMask(!matrix.isTransposed() ? getMask().reference() : getMask().transpose());
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
     * Updates slice dimensions.
     *
     * @param startRow slice start row
     * @param startColumn slice start columns
     * @param endRow slice end row
     * @param endColumn slide end column
     */
    protected void updateSliceDimensions(int startRow, int startColumn, int endRow, int endColumn) {
        sliceStartRow = startRow;
        sliceStartColumn = startColumn;
        sliceRows = endRow - sliceStartRow + 1;
        sliceColumns = endColumn - sliceStartColumn + 1;
        sliceSize = sliceRows * sliceColumns;
    }

    /**
     * Returns slice start row.
     *
     * @return slice start row.
     */
    protected int getSliceStartRow() {
        return sliceStartRow;
    }

    /**
     * Returns slice start column.
     *
     * @return slice start column.
     */
    protected int getSliceStartColumn() {
        return sliceStartColumn;
    }

    /**
     * Returns number of slice rows.
     *
     * @return number of slice rows.
     */
    protected int getSliceRows() {
        return sliceRows;
    }

    /**
     * Returns number of slice columns.
     *
     * @return number of slice columns.
     */
    protected int getSliceColumns() {
        return sliceColumns;
    }

    /**
     * Slices matrix.
     *
     * @param startRow start row of slice.
     * @param startColumn start column of slice.
     * @param endRow  end row of slice.
     * @param endColumn  end column of slice.
     * @throws MatrixException throws exception if slicing fails.
     */
    public void slice(int startRow, int startColumn, int endRow, int endColumn) throws MatrixException {
        if (startRow < 0 || startColumn < 0 || (!isTransposed() ? endRow : endColumn) > getPureRows() -1 || (!isTransposed() ? endColumn : endRow) > getPureColumns() - 1) {
            throw new MatrixException("Slice rows: " + startRow + " - " + endRow + " and slice columns: " + startColumn + " - " + endColumn + " do not match matrix dimensions: " + getTotalRows() + "x" + getTotalColumns());
        }
        else updateSliceDimensions(startRow, startColumn, (!isTransposed() ? endRow : endColumn), (!isTransposed() ? endColumn : endRow));
    }

    /**
     * Removes slicing of matrix.
     *
     */
    public void unslice() {
        updateSliceDimensions(0, 0, getPureRows() - 1, getPureColumns() - 1);
    }

    /**
     * Returns size (rows * columns) of matrix
     *
     * @return size of matrix.
     */
    public int size() {
        return sliceSize;
    }

    /**
     * Returns number of rows in matrix.
     *
     * @return number of rows in matrix.
     */
    public int getRows() {
        return !isTransposed() ? getSliceRows() : getSliceColumns();
    }

    /**
     * Returns number of columns in matrix.
     *
     * @return number of columns in matrix.
     */
    public int getColumns() {
        return !isTransposed() ? getSliceColumns() : getSliceRows();
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
     * Copies new matrix data into this matrix. Assumes equal dimensions for both matrices.
     *
     * @param newMatrix new matrix to be copied inside this matrix.
     * @throws MatrixException throws exception if this and new matrix dimensions are not matching.
     */
    public void copyMatrixData(Matrix newMatrix) throws MatrixException {
        if (getPureRows() != newMatrix.getRows() || getPureColumns() != newMatrix.getColumns()) throw new MatrixException("Size of this matrix " + getPureRows() + "x" + getPureColumns() + " is not matching with dimensions of new matrix + " + newMatrix.getRows() + " x " + newMatrix.getColumns());
        resetMatrix();
        int rows = getPureRows();
        int columns = getPureColumns();
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                setValue(row, column, newMatrix.getValue(row, column));
            }
        }
    }

    /**
     * Checks if this matrix and other matrix are equal in dimensions (rows x columns).
     *
     * @param other other matrix to be compared against.
     * @return true if matrices are of same size otherwise false.
     */
    public boolean hasEqualSize(Matrix other) {
        return other.getRows() == getRows() && other.getColumns() == getColumns();
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
     * Returns matrix of given size (rows x columns)
     *
     * @param rows rows
     * @param columns columns
     * @return new matrix
     */
    protected abstract Matrix getNewMatrix(int rows, int columns);

    /**
     * Returns constant matrix
     *
     * @param constant constant
     * @return new matrix
     */
    protected abstract Matrix getNewMatrix(double constant);

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
     */
    public Matrix getNewMatrix(boolean asTransposed) {
        return isScalar() ? getNewMatrix(0) : !asTransposed ? getNewMatrix(getRows(), getColumns()) :  getNewMatrix(getColumns(), getRows());
    }

    /**
     * Returns placeholder for result matrix.
     *
     * @param other other matrix.
     * @return result matrix placeholder.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    private Matrix getResultMatrix(Matrix other) throws MatrixException {
        return !isScalar() ? getNewMatrix() : other.getNewMatrix();
    }

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param unaryFunction unary function.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public void apply(UnaryFunction unaryFunction, boolean inplace) throws MatrixException {
        apply(inplace ? this : getNewMatrix(unaryFunction.getType() == UnaryFunctionType.TRANSPOSE), unaryFunction);
    }

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param result result matrix.
     * @param unaryFunction unaryFunction to be applied.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(Matrix result, UnaryFunction unaryFunction) throws MatrixException {
        if (!hasProcedureFactory()) applyFunction(result, unaryFunction);
        else {
            ProcedureFactory.synchronize(this, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyFunction(result, unaryFunction);
            procedureFactory.createUnaryFunctionExpression(expressionLock, this, result, unaryFunction);
        }
    }
    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param result result.
     * @param unaryFunction unary function.
     * @throws MatrixException not thrown in any situation.
     */
    protected abstract void applyFunction(Matrix result, UnaryFunction unaryFunction) throws MatrixException;

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
        Matrix result = unaryFunction.getType() != UnaryFunctionType.TRANSPOSE ? getNewMatrix() : getNewMatrix().transpose();
        apply(result, unaryFunction);
        return result;
    }

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param result result matrix.
     * @param unaryFunctionType unaryFunction type to be applied.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void apply(Matrix result, UnaryFunctionType unaryFunctionType) throws MatrixException, DynamicParamException {
        apply(result, new UnaryFunction(unaryFunctionType));
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
        return apply(new UnaryFunction(unaryFunctionType));
    }

    /**
     * Applies two variable operation to this matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other other matrix
     * @param result result matrix.
     * @param binaryFunction binaryFunction to be applied.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyBi(Matrix other, Matrix result, BinaryFunction binaryFunction) throws MatrixException {
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) applyBiFunction(other, result, binaryFunction);
        else {
            ProcedureFactory.synchronize(this, other, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyBiFunction(other, result, binaryFunction);
            procedureFactory.createBinaryFunctionExpression(expressionLock, this, other, result, binaryFunction);
        }
    }

    /**
     * Applies two variable operation to this matrix and other matrix and stores operation result into result matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix or
     * multiplying current matrix with other matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @param binaryFunction binary function.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     */
    protected abstract void applyBiFunction(Matrix other, Matrix result, BinaryFunction binaryFunction) throws MatrixException;

    /**
     * Applies two variable operation to this matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other other matrix
     * @param binaryFunction binaryFunction to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix applyBi(Matrix other, BinaryFunction binaryFunction) throws MatrixException {
        Matrix result = getResultMatrix(other);
        applyBi(other, result, binaryFunction);
        return result;
    }

    /**
     * Applies two variable operation to this matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other other matrix
     * @param result result matrix.
     * @param binaryFunctionType binaryFunction type to be applied.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix applyBi(Matrix other, Matrix result, BinaryFunctionType binaryFunctionType) throws MatrixException, DynamicParamException {
        applyBi(other, result, new BinaryFunction(binaryFunctionType));
        return result;
    }

    /**
     * Applies two variable operation to this matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other other matrix
     * @param binaryFunctionType binaryFunction type to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix applyBi(Matrix other, BinaryFunctionType binaryFunctionType) throws MatrixException, DynamicParamException {
        return applyBi(other, new BinaryFunction(binaryFunctionType));
    }

    /**
     * Adds other matrix to this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void add(Matrix other, Matrix result) throws MatrixException {
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) applyBi (other, result, new BinaryFunction((Matrix.MatrixBinaryOperation & Serializable) Double::sum));
        else {
            ProcedureFactory.synchronize(this, other, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyBi (other, result, new BinaryFunction((Matrix.MatrixBinaryOperation & Serializable) Double::sum));
            procedureFactory.createAddExpression(expressionLock, this, other, result);
        }
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
        Matrix result = getResultMatrix(other);
        add(other, result);
        return result;
    }

    /**
     * Adds constant number to this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant contains constant value to be added.
     * @param result matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public void add(double constant, Matrix result) throws MatrixException {
        Matrix other = getNewMatrix(constant);
        if (!hasProcedureFactory()) add(other, result);
        else {
            ProcedureFactory.synchronize(this, other, result);
            int expressionLock = procedureFactory.startExpression(this);
            add(other, result);
            procedureFactory.createAddExpression(expressionLock, this, other, result);
        }
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
        Matrix result = getNewMatrix();
        add(constant, result);
        return result;
    }

    /**
     * Subtracts other matrix from this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void subtract(Matrix other, Matrix result) throws MatrixException {
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) applyBi (other, result, new BinaryFunction((Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 - value2));
        else {
            ProcedureFactory.synchronize(this, other, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyBi (other, result, new BinaryFunction((Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 - value2));
            procedureFactory.createSubtractExpression(expressionLock, this, other, result);
        }
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
        Matrix result = getResultMatrix(other);
        subtract(other, result);
        return result;
    }

    /**
     * Subtracts constant number from this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant contains constant value to be subtracted.
     * @param result matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public void subtract(double constant, Matrix result) throws MatrixException {
        Matrix other = getNewMatrix(constant);
        if (!hasProcedureFactory()) subtract(other, result);
        else {
            ProcedureFactory.synchronize(this, other, result);
            int expressionLock = procedureFactory.startExpression(this);
            subtract(other, result);
            procedureFactory.createSubtractExpression(expressionLock, this, other, result);
        }
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
        Matrix result = getNewMatrix();
        subtract(constant, result);
        return result;
    }

    /**
     * Multiplies other matrix element wise with this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void multiply(Matrix other, Matrix result) throws MatrixException {
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) applyBi (other, result, new BinaryFunction((Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 * value2));
        else {
            ProcedureFactory.synchronize(this, other, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyBi (other, result, new BinaryFunction((Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 * value2));
            procedureFactory.createMultiplyExpression(expressionLock, this, other, result);
        }
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
        Matrix result = getResultMatrix(other);
        multiply(other, result);
        return result;
    }

    /**
     * Multiplies constant number with this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant contains constant value to be multiplied.
     * @param result matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public void multiply(double constant, Matrix result) throws MatrixException {
        Matrix other = getNewMatrix(constant);
        if (!hasProcedureFactory()) multiply(other, result);
        else {
            ProcedureFactory.synchronize(this, other, result);
            int expressionLock = procedureFactory.startExpression(this);
            multiply(other, result);
            procedureFactory.createMultiplyExpression(expressionLock, this, other, result);
        }
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
        Matrix result = getNewMatrix();
        multiply(constant, result);
        return result;
    }

    /**
     * Divides this matrix element wise with other matrix.<br>
     * In case any element value of other matrix is zero result is treated as Double MAX value to avoid NaN condition.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     */
    public void divide(Matrix other, Matrix result) throws MatrixException {
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) applyBi (other, result, new BinaryFunction((Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value2 != 0 ? value1 / value2 : Double.POSITIVE_INFINITY));
        else {
            ProcedureFactory.synchronize(this, other, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyBi (other, result, new BinaryFunction((Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 / value2));
            procedureFactory.createDivideExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Divides this matrix element wise with other matrix.<br>
     * In case any element value of other matrix is zero result is treated as Double MAX value to avoid NaN condition.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public Matrix divide(Matrix other) throws MatrixException {
        Matrix result = getResultMatrix(other);
        divide(other, result);
        return result;
    }

    /**
     * Divides this matrix element wise with constant.<br>
     * In case constant is zero result is treated as Double MAX value to avoid NaN condition.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant constant used as divider value.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and result matrix are not of equal dimensions.
     */
    public void divide(double constant, Matrix result) throws MatrixException {
        Matrix other = getNewMatrix(constant);
        if (!hasProcedureFactory()) divide(other, result);
        else {
            ProcedureFactory.synchronize(this, other, result);
            int expressionLock = procedureFactory.startExpression(this);
            divide(other, result);
            procedureFactory.createDivideExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Divides this matrix element wise with constant.<br>
     * In case constant is zero result is treated as Double MAX value to avoid NaN condition.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant constant used as divider value.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix divide(double constant) throws MatrixException {
        Matrix result = getNewMatrix();
        divide(constant, result);
        return result;
    }

    /**
     * Increments matrix by other matrix.
     *
     * @param otherMatrix other matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void incrementBy(Matrix otherMatrix) throws MatrixException {
        add(otherMatrix, this);
    }

    /**
     * Decrements matrix by other matrix.
     *
     * @param otherMatrix other matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void decrementBy(Matrix otherMatrix) throws MatrixException {
        subtract(otherMatrix, this);
    }

    /**
     * Multiplies matrix by other matrix.
     *
     * @param otherMatrix other matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void multiplyBy(Matrix otherMatrix) throws MatrixException {
        multiply(otherMatrix, this);
    }

    /**
     * Divides matrix by other matrix.
     *
     * @param otherMatrix other matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void divideBy(Matrix otherMatrix) throws MatrixException {
        divide(otherMatrix, this);
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
        return applyBi (constantAsMatrix(power), BinaryFunctionType.POW);
    }

    /**
     * Raises this matrix element wise to the power of value power.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param power power value to which elements are to be raised.
     * @param result matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void power(double power, Matrix result) throws MatrixException, DynamicParamException {
        applyBi (constantAsMatrix(power), result, BinaryFunctionType.POW);
    }

    /**
     * Takes element wise max value of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void max(Matrix other, Matrix result) throws MatrixException, DynamicParamException {
        applyBi (other, result, BinaryFunctionType.MAX);
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
     * Takes element wise min value of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void min(Matrix other, Matrix result) throws MatrixException, DynamicParamException {
        applyBi (other, result, BinaryFunctionType.MIN);
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
     * Takes element wise signum over multiplication of this and other matrix.<br>
     * Applies first sign operation to each value and then multiplies them.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void sgnmul(Matrix other, Matrix result) throws MatrixException, DynamicParamException {
        apply(UnaryFunctionType.SGN).multiply(other.apply(UnaryFunctionType.SGN), result);
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
     * Takes matrix dot product of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if columns of this matrix and rows of other matrix are not matching or rows of this and result matrix or columns of result and other matrix are not matching.
     */
    public Matrix dot(Matrix other, Matrix result) throws MatrixException {
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) applyDot(other, result);
        else {
            ProcedureFactory.synchronize(this, other, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyDot(other, result);
            procedureFactory.createDotExpression(expressionLock, this, other, result);
        }
        return result;
    }

    /**
     * Takes matrix dot product of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if columns of this matrix and rows of other matrix are not matching or rows of this and result matrix or columns of result and other matrix are not matching.
     */
    protected abstract void applyDot(Matrix other, Matrix result) throws MatrixException;

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
        return dot(other, getNewMatrix(getRows(), other.getColumns()));
    }

    /**
     * Returns constant as matrix.
     *
     * @param constant constant value.
     * @return constant matrix.
     */
    public Matrix constantAsMatrix(double constant) {
        Matrix constantMatrix = getNewMatrix(constant);
        constantMatrix.setValue(0, 0, constant);
        return constantMatrix;
    }

    /**
     * Takes element wise cumulative sum of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return sum of matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix sumAsMatrix() throws MatrixException {
        if (!hasProcedureFactory()) return constantAsMatrix(sum());
        else {
            int expressionLock = procedureFactory.startExpression(this);
            Matrix result = constantAsMatrix(sum());
            ProcedureFactory.synchronize(this, result);
            procedureFactory.createSumExpression(expressionLock, this, result);
            return result;
        }
    }

    /**
     * Takes mean of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @throws MatrixException not thrown in any situation.
     * @return mean of matrix.
     */
    public Matrix meanAsMatrix() throws MatrixException {
        if (!hasProcedureFactory()) return constantAsMatrix(mean());
        else {
            int expressionLock = procedureFactory.startExpression(this);
            Matrix result = constantAsMatrix(mean());
            ProcedureFactory.synchronize(this, result);
            procedureFactory.createMeanExpression(expressionLock, this, result);
            return result;
        }
    }

    /**
     * Takes variance of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return variance of matrix.
     */
    public double variance() throws MatrixException {
        return variance(mean());
    }

    /**
     * Takes variance of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @throws MatrixException not thrown in any situation.
     * @return variance of matrix.
     */
    public Matrix varianceAsMatrix() throws MatrixException {
        if (!hasProcedureFactory()) return constantAsMatrix(variance());
        else {
            int expressionLock = procedureFactory.startExpression(this);
            Matrix result = constantAsMatrix(variance());
            ProcedureFactory.synchronize(this, result);
            procedureFactory.createVarianceExpression(expressionLock, this, result);
            return result;
        }
    }

    /**
     * Takes variance of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return variance of matrix.
     */
    public Matrix varianceAsMatrix(Matrix mean) throws MatrixException {
        return constantAsMatrix(variance(mean.getValue(0, 0)));
    }

    /**
     * Takes standard deviation of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return standard deviation of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double standardDeviation() throws MatrixException {
        return standardDeviation(mean());
    }

    /**
     * Takes standard deviation of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @throws MatrixException not thrown in any situation.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @return standard deviation of matrix.
     */
    public Matrix standardDeviationAsMatrix() throws MatrixException, DynamicParamException {
        if (!hasProcedureFactory()) return constantAsMatrix(standardDeviation());
        else {
            int expressionLock = procedureFactory.startExpression(this);
            Matrix result = constantAsMatrix(standardDeviation());
            ProcedureFactory.synchronize(this, result);
            procedureFactory.createStandardDeviationExpression(expressionLock, this, result);
            return result;
        }
    }

    /**
     * Takes standard deviation of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return standard deviation of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix standardDeviationAsMatrix(Matrix mean) throws MatrixException {
        return constantAsMatrix(standardDeviation(mean.getValue(0, 0)));
    }

    /**
     * Takes cumulative p- norm (p is number equal or bigger than 1) of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param p p value for norm.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return norm of matrix.
     */
    public Matrix normAsMatrix(int p) throws MatrixException {
        if (!hasProcedureFactory()) return constantAsMatrix(norm(p));
        else {
            int expressionLock = procedureFactory.startExpression(this);
            Matrix result = constantAsMatrix(norm(p));
            ProcedureFactory.synchronize(this, result);
            procedureFactory.createNormExpression(expressionLock, this, result, p);
            return result;
        }
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
     * @return minimum value of matrix.
     */
    public Matrix minAsMatrix() throws MatrixException {
        return constantAsMatrix(min());
    }

    /**
     * Returns maximum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return maximum value of matrix.
     */
    public Matrix maxAsMatrix() throws MatrixException {
        return constantAsMatrix(max());
    }

    /**
     * Returns entropy of matrix.
     *
     * @param asDistribution if true matrix is forced into distribution prior calculating entropy.
     * @return entropy of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double entropy(boolean asDistribution) throws MatrixException {
        return divide(sum()).entropy();
    }

    /**
     * Returns softmax of this matrix.
     *
     * @return softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmax() throws MatrixException {
        return softmax(getNewMatrix());
    }

    /**
     * Returns Gumbel softmax of this matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @param result result matrix.
     * @return Gumbel softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix gumbelSoftmax(Matrix result) throws MatrixException {
        return gumbelSoftmax(result, 1);
    }

    /**
     * Returns Gumbel softmax of this matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @return Gumbel softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix gumbelSoftmax() throws MatrixException {
        return gumbelSoftmax(getNewMatrix(), 1);
    }

    /**
     * Returns Gumbel softmax of this matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @param gumbelSoftmaxTau tau value for Gumbel Softmax.
     * @return Gumbel softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix gumbelSoftmax(double gumbelSoftmaxTau) throws MatrixException {
        return gumbelSoftmax(getNewMatrix(), gumbelSoftmaxTau);
    }

    /**
     * Returns softmax gradient of this matrix.
     *
     * @return softmax gradient of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmaxGrad() throws MatrixException {
        return softmaxGrad(getNewMatrix(getRows(), getRows()));
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated result of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix convolve(Matrix filter) throws MatrixException {
        Matrix result = getNewMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1);
        convolve(filter, result);
        return result;
    }

    /**
     * Calculates crosscorrelation between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated result of crosscorrelation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix crosscorrelate(Matrix filter) throws MatrixException {
        Matrix result = getNewMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1);
        crosscorrelate(filter, result);
        return result;
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated result of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void convolve(Matrix filter, Matrix result) throws MatrixException {
        if (!hasProcedureFactory() && !filter.hasProcedureFactory()) applyConvolve(filter, result);
        else {
            ProcedureFactory.synchronize(this, filter, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyConvolve(filter, result);
            procedureFactory.createConvolveExpression(expressionLock, this, filter, result, getStride(), getDilation());
        }
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated result of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract void applyConvolve(Matrix filter, Matrix result) throws MatrixException;

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated result of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void crosscorrelate(Matrix filter, Matrix result) throws MatrixException {
        if (!hasProcedureFactory() && !filter.hasProcedureFactory()) applyCrosscorrelate(filter, result);
        else {
            ProcedureFactory.synchronize(this, filter, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyCrosscorrelate(filter, result);
            procedureFactory.createCrosscorrelateExpression(expressionLock, this, filter, result, getStride(), getDilation());
        }
    }

    /**
     * Calculates crosscorrelation between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated result of crosscorrelation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract void applyCrosscorrelate(Matrix filter, Matrix result) throws MatrixException;

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated value of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix winogradConvolve(Matrix filter) throws MatrixException {
        Matrix result = getNewMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1);
        winogradConvolve(filter, result);
        return result;
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated value of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void winogradConvolve(Matrix filter, Matrix result) throws MatrixException {
        if (!hasProcedureFactory() && !filter.hasProcedureFactory()) applyWinogradConvolve(filter, result);
        else {
            ProcedureFactory.synchronize(this, filter, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyWinogradConvolve(filter, result);
            procedureFactory.createWinogradConvolveExpression(expressionLock, this, filter, result, getStride(), getDilation());
        }
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
     */
    public Matrix winogradConvolve(Matrix filter, Matrix A, Matrix AT, Matrix C, Matrix CT, Matrix G, Matrix GT) throws MatrixException {
        Matrix result = getNewMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1);
        winogradConvolve(filter, result, A, AT, C, CT, G, GT);
        return result;
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated value of convolution.
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     * @param G G matrix
     * @param GT G transposed matrix
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void winogradConvolve(Matrix filter, Matrix result, Matrix A, Matrix AT, Matrix C, Matrix CT, Matrix G, Matrix GT) throws MatrixException {
        if (!hasProcedureFactory() && !filter.hasProcedureFactory()) applyWinogradConvolve(filter, result, A, AT, C, CT, G, GT);
        else {
            ProcedureFactory.synchronize(this, filter, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyWinogradConvolve(filter, result, A, AT, C, CT, G, GT);
            procedureFactory.createWinogradConvolveExpression(expressionLock, this, filter, result, getStride(), getDilation());
        }
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
     */
    public Matrix winogradConvolve(Matrix preprocessedFilter, Matrix A, Matrix AT, Matrix C, Matrix CT) throws MatrixException {
        Matrix result = getNewMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1);
        winogradConvolve(preprocessedFilter, result, A, AT, C, CT);
        return result;
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param preprocessedFilter preprocessed filter matrix.
     * @param result calculated value of convolution.
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void winogradConvolve(Matrix preprocessedFilter, Matrix result, Matrix A, Matrix AT, Matrix C, Matrix CT) throws MatrixException {
        if (!hasProcedureFactory() && !preprocessedFilter.hasProcedureFactory()) applyWinogradConvolve(preprocessedFilter, result, A, AT, C, CT);
        else {
            ProcedureFactory.synchronize(this, preprocessedFilter, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyWinogradConvolve(preprocessedFilter, result, A, AT, C, CT);
            procedureFactory.createWinogradConvolveExpression(expressionLock, this, preprocessedFilter, result, getStride(), getDilation());
        }
    }

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated result of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract void applyWinogradConvolve(Matrix filter, Matrix result) throws MatrixException;

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated result of convolution.
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     * @param G G matrix
     * @param GT G transposed matrix
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract void applyWinogradConvolve(Matrix filter, Matrix result, Matrix A, Matrix AT, Matrix C, Matrix CT, Matrix G, Matrix GT) throws MatrixException;

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param preprocessedFilter preprocessed filter matrix.
     * @param result calculated result of convolution.
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract void applyWinogradConvolve(Matrix preprocessedFilter, Matrix result, Matrix A, Matrix AT, Matrix C, Matrix CT) throws MatrixException;

    /**
     * Calculates gradient of convolution for input.
     *
     * @param filter filter for convolutional operator.
     * @return input gradient.
     */
    public Matrix convolveInputGradient(Matrix filter) throws MatrixException {
        Matrix inputGradient = getNewMatrix(getRows() + getFilterRowSize() - 1, getColumns() + getFilterColumnSize() - 1);
        convolveInputGradient(filter, inputGradient);
        return inputGradient;
    }

    /**
     * Calculates gradient of crosscorrelation for input.
     *
     * @param filter filter for crosscorrelation operator.
     * @return input gradient.
     */
    public Matrix crosscorrelateInputGradient(Matrix filter) throws MatrixException {
        Matrix inputGradient = getNewMatrix(getRows() + getFilterRowSize() - 1, getColumns() + getFilterColumnSize() - 1);
        crosscorrelateInputGradient(filter, inputGradient);
        return inputGradient;
    }

    /**
     * Calculates gradient of convolution for filter.
     *
     * @param input input for convolutional operator.
     * @return filter gradient.
     */
    public Matrix convolveFilterGradient(Matrix input) throws MatrixException {
        Matrix filterGradient = getNewMatrix(getFilterRowSize(), getFilterColumnSize());
        convolveFilterGradient(input, filterGradient);
        return filterGradient;
    }

    /**
     * Calculates gradient of crosscorrelation for filter.
     *
     * @param input input for crosscorrelation operator.
     * @return filter gradient.
     */
    public Matrix crosscorrelateFilterGradient(Matrix input) throws MatrixException {
        Matrix filterGradient = getNewMatrix(getFilterRowSize(), getFilterColumnSize());
        crosscorrelateFilterGradient(input, filterGradient);
        return filterGradient;
    }

    /**
     * Calculates max pooling operation for this matrix.
     *
     * @param maxPos maximum positions for each row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix maxPool(HashMap<Integer, Integer> maxPos) throws MatrixException {
        Matrix result = getNewMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1);
        maxPool(result, maxPos);
        return result;
    }

    /**
     * Calculates max pooling operation for this matrix and returns max arguments.
     *
     * @param result result matrix.
     * @param maxPos maximum positions for each row and col value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void maxPool(Matrix result, HashMap<Integer, Integer> maxPos) throws MatrixException {
        if (!hasProcedureFactory()) applyMaxPool(result, maxPos);
        else {
            ProcedureFactory.synchronize(this, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyMaxPool(result, maxPos);
            procedureFactory.createMaxPoolExpression(expressionLock, this, result, getStride(), getFilterRowSize(), getFilterColumnSize());
        }
    }

    /**
     * Calculates max pooling operation for this matrix and returns max arguments.
     *
     * @param result result matrix.
     * @param maxPos maximum positions for each row and col value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract void applyMaxPool(Matrix result, HashMap<Integer, Integer> maxPos) throws MatrixException;

    /**
     * Calculates gradient of max pooling operation for this matrix.
     *
     * @param maxPos maximum positions for each row and col value.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix maxPoolGradient(HashMap<Integer, Integer> maxPos) throws MatrixException {
        Matrix inputGradient = getNewMatrix(getRows() + getFilterRowSize() - 1, getColumns() + getFilterColumnSize() - 1);
        maxPoolGradient(inputGradient, maxPos);
        return inputGradient;
    }

    /**
     * Calculates random pooling operation for this matrix.
     *
     * @param inputPos input positions for each row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix randomPool(HashMap<Integer, Integer> inputPos) throws MatrixException {
        Matrix result = getNewMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1);
        randomPool(result, inputPos);
        return result;
    }

    /**
     * Calculates random pooling operation for this matrix and returns input positions.
     *
     * @param result result matrix.
     * @param inputPos input positions for each row and col value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void randomPool(Matrix result, HashMap<Integer, Integer> inputPos) throws MatrixException {
        if (!hasProcedureFactory()) applyMaxPool(result, inputPos);
        else {
            ProcedureFactory.synchronize(this, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyRandomPool(result, inputPos);
            procedureFactory.createRandomPoolExpression(expressionLock, this, result, getStride(), getFilterRowSize(), getFilterColumnSize());
        }
    }

    /**
     * Calculates random pooling operation for this matrix and returns input positions.
     *
     * @param result result matrix.
     * @param inputPos input positions for each row and col value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract void applyRandomPool(Matrix result, HashMap<Integer, Integer> inputPos) throws MatrixException;

    /**
     * Calculates gradient of random pooling operation for this matrix.
     *
     * @param inputPos input positions for each row and col value.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix randomPoolGradient(HashMap<Integer, Integer> inputPos) throws MatrixException {
        Matrix inputGradient = getNewMatrix(getRows() + getFilterRowSize() - 1, getColumns() + getFilterColumnSize() - 1);
        randomPoolGradient(inputGradient, inputPos);
        return inputGradient;
    }

    /**
     * Calculates cyclic pooling operation for this matrix.
     *
     * @param inputPos input positions for each row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix cyclicPool(HashMap<Integer, Integer> inputPos) throws MatrixException {
        Matrix result = getNewMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1);
        cyclicPool(result, inputPos);
        return result;
    }

    /**
     * Calculates cyclic pooling operation for this matrix and returns input positions.
     *
     * @param result result matrix.
     * @param inputPos input positions for each row and col value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void cyclicPool(Matrix result, HashMap<Integer, Integer> inputPos) throws MatrixException {
        if (!hasProcedureFactory()) applyMaxPool(result, inputPos);
        else {
            ProcedureFactory.synchronize(this, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyCyclicPool(result, inputPos);
            procedureFactory.createCyclicPoolExpression(expressionLock, this, result, getStride(), getFilterRowSize(), getFilterColumnSize());
        }
    }

    /**
     * Calculates cyclic pooling operation for this matrix and returns input positions.
     *
     * @param result result matrix.
     * @param inputPos input positions for each row and col value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract void applyCyclicPool(Matrix result, HashMap<Integer, Integer> inputPos) throws MatrixException;

    /**
     * Calculates gradient of cyclic pooling operation for this matrix.
     *
     * @param inputPos input positions for each row and col value.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix cyclicPoolGradient(HashMap<Integer, Integer> inputPos) throws MatrixException {
        Matrix inputGradient = getNewMatrix(getRows() + getFilterRowSize() - 1, getColumns() + getFilterColumnSize() - 1);
        cyclicPoolGradient(inputGradient, inputPos);
        return inputGradient;
    }

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix averagePool() throws MatrixException {
        Matrix result = getNewMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1);
        averagePool(result);
        return result;
    }

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void averagePool(Matrix result) throws MatrixException {
        if (!hasProcedureFactory()) applyAveragePool(result);
        else {
            ProcedureFactory.synchronize(this, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyAveragePool(result);
            procedureFactory.createAveragePoolExpression(expressionLock, this, result, getStride(), getFilterRowSize(), getFilterColumnSize());
        }
    }

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @param inputGradient input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract void applyAveragePool(Matrix inputGradient) throws MatrixException;

    /**
     * Calculates gradient of average pooling operation for this matrix.
     *
     * @return input gradient.
     */
    public Matrix averagePoolGradient() throws MatrixException {
        Matrix inputGradient = getNewMatrix(getRows() + getFilterRowSize() - 1, getColumns() + getFilterColumnSize() - 1);
        averagePoolGradient(inputGradient);
        return inputGradient;
    }


    /**
     * Transposes matrix.
     *
     * @return transposed matrix.
     * @throws MatrixException throws exception if cloning of mask fails.
     */
    public Matrix transpose() throws MatrixException {
        if (!hasProcedureFactory()) return applyTranspose();
        else {
            Matrix result = applyTranspose();
            ProcedureFactory.synchronize(this, result);
            return result;
        }
    }

    /**
     * Applies matrix transpose.
     *
     * @return transposed matrix.
     * @throws MatrixException throws exception if cloning of mask fails.
     */
    protected abstract Matrix applyTranspose() throws MatrixException;

    /**
     * Checks if matrix is transposed.
     *
     * @return true is matrix is transposed otherwise false.
     */
    public boolean isTransposed() {
        return isTransposed;
    }

    /**
     * Splits matrix at defined position. If splitVertical is true splits vertically otherwise horizontally.
     *
     * @param position position of split
     * @param splitVertically if true splits vertically otherwise horizontally.
     * @return split matrix as JMatrix.
     * @throws MatrixException throws matrix exception if splitting fails.
     *
     */
    public Matrix split(int position, boolean splitVertically) throws MatrixException {
        if (!((this instanceof DMatrix) || (this instanceof SMatrix))) throw new MatrixException("Matrix must be of type DMatrix or SMatrix");
        Matrix matrix1;
        Matrix matrix2;
        int rows = getTotalRows();
        int columns = getTotalColumns();
        if (splitVertically) {
            if (position < 1 || position > rows - 1) throw new MatrixException("For vertical split position is beyond number of rows in matrix.");
            matrix1 = getNewMatrix(position, columns);
            matrix2 = getNewMatrix(rows - matrix1.getTotalRows(), columns);
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    if (row < position) matrix1.setValue(row, column, getValue(row, column));
                    else matrix2.setValue(row - position, column, getValue(row, column));
                }
            }
        }
        else {
            if (position < 1 || position > columns - 1) throw new MatrixException("For vertical split position is beyond number of rows in matrix.");
            matrix1 = getNewMatrix(rows, position);
            matrix2 = getNewMatrix(rows, columns - matrix1.getTotalColumns());
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    if (column < position) matrix1.setValue(row, column, getValue(row, column));
                    else matrix2.setValue(row, column - position, getValue(row, column));
                }
            }
        }
        ArrayList<Matrix> matrices = new ArrayList<>();
        matrices.add(matrix1);
        matrices.add(matrix2);
        Matrix result = new JMatrix(matrices, splitVertically);
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
            int expressionLock = procedureFactory.startExpression(this);
            Matrix result = applyJoin(other, joinedVertically);
            ProcedureFactory.synchronize(this, other, result);
            procedureFactory.createJoinExpression(expressionLock, this, other, result, joinedVertically);
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
        return new JoinMatrixOperation(getRows() + other.getRows(), getColumns() + other.getColumns(), joinedVertically).apply(this, other);
    }

    /**
     * Unjoins matrix at specific row and column.
     *
     * @param unjoinAtRow unjoins at row.
     * @param unjoinAtColumn unjoins at column.
     * @param unjoinRows unjoins specific number of rows.
     * @param unjoinColumns unjoins specific number of column.
     * @return result matrix.
     * @throws MatrixException throws matrix exception if unjoining fails.
     */
    public Matrix unjoin(int unjoinAtRow, int unjoinAtColumn, int unjoinRows, int unjoinColumns) throws MatrixException {
        Matrix result = new DMatrix(unjoinRows, unjoinColumns);
        unjoin(result, unjoinAtRow, unjoinAtColumn);
        return result;
    }

    /**
     * Unjoins matrix at specific row and column.
     *
     * @param result result matrix
     * @param unjoinAtRow unjoins at row.
     * @param unjoinAtColumn unjoins at column.
     * @throws MatrixException throws matrix exception if unjoining fails.
     */
    public void unjoin(Matrix result, int unjoinAtRow, int unjoinAtColumn) throws MatrixException {
        if (!hasProcedureFactory()) applyUnjoin(result, unjoinAtRow, unjoinAtColumn);
        else {
            ProcedureFactory.synchronize(this, result);
            int expressionLock = procedureFactory.startExpression(this);
            applyUnjoin(result, unjoinAtRow, unjoinAtColumn);
            procedureFactory.createUnjoinExpression(expressionLock, this, result, unjoinAtRow, unjoinAtColumn);
        }
    }

    /**
     * Unjoins matrix into resulting unjoined matrix and potentially unjoined matrices.
     *
     * @param result result matrix
     * @param unjoinAtRow unjoins at row.
     * @param unjoinAtColumn unjoins at column.
     * @throws MatrixException throws matrix exception if unjoining fails.
     */
    private void applyUnjoin(Matrix result, int unjoinAtRow, int unjoinAtColumn) throws MatrixException {
        new UnjoinMatrixOperation(result.getRows(), result.getColumns(), unjoinAtRow, unjoinAtColumn).apply(this, result);
    }

    /**
     * Prints matrix in row and column format.
     *
     */
    public void print() {
        int rows = getRows();
        int columns = getColumns();
        for (int row = 0; row < rows; row++) {
            System.out.print("[");
            for (int column = 0; column < columns; column++) {
                System.out.print(getValue(row, column));
                if (column < columns - 1) System.out.print(" ");
            }
            System.out.println("]");
        }
    }

    /**
     * Prints size (rows x columns) of matrix.
     *
     */
    public void printSize() {
        System.out.println("Matrix size: " + getRows() + "x" + getColumns());
    }

    /**
     * Sets mask to this matrix.
     *
     * @param newMask new mask as input.
     * @throws MatrixException throws exception if new mask dimensions or mask type are not matching with this mask.
     */
    public void setMask(Mask newMask) throws MatrixException {
        if (getRows() != newMask.getRows() || getColumns() != newMask.getColumns()) throw new MatrixException("Dimensions of new mask are not matching with matrix dimensions.");
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
     * @return if true mask exists and is masked at specific position (row + column).
     */
    public boolean hasMaskAt(int row, int column) {
        return getMask() != null && getMask().getMask(row, column);
    }

    /**
     * Returns new mask for this matrix.<br>
     * Implemented by underlying matrix class.<br>
     *
     * @return mask of this matrix.
     */
    protected abstract Mask getNewMask();

}
