/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
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
     * Constructor for matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     */
    protected AbstractMatrix(int rows, int columns, int depth) {
        this(rows, columns, depth, false);
    }

    /**
     * Constructor for abstract matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param isTransposed if true matrix is transposed and if false not transposed.
     */
    protected AbstractMatrix(int rows, int columns, int depth, boolean isTransposed) {
        this(rows, columns, depth, isTransposed, false);
    }

    /**
     * Constructor for abstract matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param isTransposed if true matrix is transposed and if false not transposed.
     * @param canBeSliced if true matrix can be slides otherwise cannot be sliced.
     */
    protected AbstractMatrix(int rows, int columns, int depth, boolean isTransposed, boolean canBeSliced) {
        this.rows = rows;
        this.columns = columns;
        this.depth = depth;
        this.isTransposed = isTransposed;
        this.canBeSliced = canBeSliced;
        if (canBeSliced()) updateSliceDimensions(0, 0, 0, rows - 1, columns - 1, depth -1);
    }

    /**
     * Sets parameters for matrix.
     *
     * @param matrix matrix.
     * @throws MatrixException throws exception if cloning of mask fails.
     */
    protected void setParameters(Matrix matrix) throws MatrixException {
        matrix.setName(name);
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
     * @throws MatrixException throws exception if new mask dimensions or mask type are not matching with this mask.
     */
    public Matrix getNewMatrix(boolean asTransposed) throws MatrixException {
        return isScalar() ? getNewMatrix(0) : !asTransposed ? getNewMatrix(getRows(), getColumns(), getDepth()) :  getNewMatrix(getColumns(), getRows(), getDepth());
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
            int expressionLock = getProcedureFactory().startExpression(this);
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
    protected abstract Matrix applyFunction(UnaryFunction unaryFunction, boolean inplace) throws MatrixException;

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
            int expressionLock = getProcedureFactory().startExpression(this);
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
    protected abstract Matrix applyBiFunction(Matrix other, BinaryFunction binaryFunction, boolean inplace) throws MatrixException;

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
            int expressionLock = getProcedureFactory().startExpression(this);
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
            int expressionLock = firstMatrix.getProcedureFactory().startExpression(firstMatrix);
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
            int expressionLock = getProcedureFactory().startExpression(this);
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
            int expressionLock = firstMatrix.getProcedureFactory().startExpression(firstMatrix);
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
            int expressionLock = getProcedureFactory().startExpression(this);
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
            int expressionLock = firstMatrix.getProcedureFactory().startExpression(firstMatrix);
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
            int expressionLock = getProcedureFactory().startExpression(this);
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
            int expressionLock = firstMatrix.getProcedureFactory().startExpression(firstMatrix);
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
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix sgnmul(Matrix other) throws MatrixException, DynamicParamException {
        return apply(UnaryFunctionType.SGN).multiply(other.apply(UnaryFunctionType.SGN));
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
            ProcedureFactory.synchronize(this, other);
            int expressionLock = getProcedureFactory().startExpression(this);
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
    protected abstract Matrix applyDot(Matrix other) throws MatrixException;

    /**
     * Returns constant as matrix.
     *
     * @param constant constant value.
     * @return constant matrix.
     */
    public Matrix constantAsMatrix(double constant) {
        Matrix constantMatrix = getNewMatrix(constant);
        constantMatrix.setValue(0, 0, 0, constant);
        return constantMatrix;
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
            int expressionLock = getProcedureFactory().startExpression(this);
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
    protected abstract Matrix applySumAsMatrix(int direction) throws MatrixException;

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
            int expressionLock = getProcedureFactory().startExpression(this);
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
    protected abstract Matrix applyMeanAsMatrix(int direction) throws MatrixException;

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
            int expressionLock = firstMatrix.getProcedureFactory().startExpression(firstMatrix);
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
            int expressionLock = firstMatrix.getProcedureFactory().startExpression(firstMatrix);
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
     */
    public double variance() throws MatrixException {
        return variance(mean());
    }

    /**
     * Takes variance of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return variance of matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix varianceAsMatrix(int direction) throws MatrixException {
        return varianceAsMatrix(null, direction);
    }

    /**
     * Calculates variance as matrix.
     *
     * @param meanMatrix mean matrix given as input.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return variance as matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract Matrix applyVarianceAsMatrix(Matrix meanMatrix, int direction) throws MatrixException;

    /**
     * Takes variance of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param meanMatrix mean matrix given as input.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return variance of matrix.
     */
    public Matrix varianceAsMatrix(Matrix meanMatrix, int direction) throws MatrixException {
        if (!hasProcedureFactory()) return applyVarianceAsMatrix(meanMatrix, direction);
        else {
            int expressionLock = getProcedureFactory().startExpression(this);
            Matrix result = applyVarianceAsMatrix(meanMatrix, direction);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createVarianceExpression(expressionLock, this, result, direction);
            return result;
        }
    }

    /**
     * Calculates variance.
     *
     * @param matrices matrices.
     * @param meanMatrix matrix containing mean values for variance calculation.
     * @return resulting variance
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static Matrix variance(TreeMap<Integer, Matrix> matrices, Matrix meanMatrix) throws MatrixException, DynamicParamException {
        Matrix firstMatrix = matrices.get(matrices.firstKey());
        if (!firstMatrix.hasProcedureFactory()) return applyVariance(matrices, meanMatrix);
        else {
            int expressionLock = firstMatrix.getProcedureFactory().startExpression(firstMatrix);
            Matrix result = applyVariance(matrices, meanMatrix);
            if (result != null) {
                ProcedureFactory.synchronize(firstMatrix, result);
            }
            firstMatrix.getProcedureFactory().createVarianceExpression(expressionLock, firstMatrix, result, true, 0);
            return result;
        }
    }

    /**
     * Calculates variance.
     *
     * @param matrices matrices.
     * @param meanMatrix matrix containing mean values for variance calculation.
     * @return resulting variance
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private static Matrix applyVariance(TreeMap<Integer, Matrix> matrices, Matrix meanMatrix) throws MatrixException, DynamicParamException {
        if (meanMatrix == null) throw new MatrixException("Mean matrix is not defined");
        Matrix result = null;
        for (Matrix matrix : matrices.values()) {
            if (result == null) result = matrix.getNewMatrix();
            result.addBy(matrix.subtract(meanMatrix).power(2));
        }
        return result == null ? null : result.divide(matrices.size());
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
     * Calculates standard deviation.
     *
     * @param matrices matrices.
     * @param meanMatrix matrix containing mean values for standard deviation calculation.
     * @return resulting standard deviation
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static Matrix standardDeviation(TreeMap<Integer, Matrix> matrices, Matrix meanMatrix) throws MatrixException, DynamicParamException {
        Matrix firstMatrix = matrices.get(matrices.firstKey());
        if (!firstMatrix.hasProcedureFactory()) return applyStandardDeviation(matrices, meanMatrix);
        else {
            int expressionLock = firstMatrix.getProcedureFactory().startExpression(firstMatrix);
            Matrix result = applyStandardDeviation(matrices, meanMatrix);
            if (result != null) {
                ProcedureFactory.synchronize(firstMatrix, result);
            }
            firstMatrix.getProcedureFactory().createStandardDeviationExpression(expressionLock, firstMatrix, result, true, 0);
            return result;
        }
    }

    /**
     * Calculates standard deviation.
     *
     * @param matrices matrices.
     * @param meanMatrix matrix containing mean values for variance calculation.
     * @return resulting standard deviation.
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private static Matrix applyStandardDeviation(TreeMap<Integer, Matrix> matrices, Matrix meanMatrix) throws MatrixException, DynamicParamException {
        if (meanMatrix == null) throw new MatrixException("Mean matrix is not defined");
        Matrix result = null;
        for (Matrix matrix : matrices.values()) {
            if (result == null) result = matrix.getNewMatrix();
            result.addBy(matrix.subtract(meanMatrix).power(2));
        }
        return result == null ? null : result.divide(matrices.size()).multiply(matrices.size()).divide(matrices.size() - 1).apply(UnaryFunctionType.SQRT);
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
     * Calculates standard deviation as matrix.
     *
     * @param meanMatrix mean value given as input.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return standard deviation as matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected abstract Matrix applyStandardDeviationAsMatrix(Matrix meanMatrix, int direction) throws MatrixException, DynamicParamException;

    /**
     * Takes standard deviation of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param meanMatrix mean value given as input.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return standard deviation of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix standardDeviationAsMatrix(Matrix meanMatrix, int direction) throws MatrixException, DynamicParamException {
        if (!hasProcedureFactory()) return applyStandardDeviationAsMatrix(meanMatrix, direction);
        else {
            int expressionLock = getProcedureFactory().startExpression(this);
            Matrix result = applyStandardDeviationAsMatrix(meanMatrix, direction);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createStandardDeviationExpression(expressionLock, this, result, direction);
            return result;
        }
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
            int expressionLock = getProcedureFactory().startExpression(this);
            Matrix result = constantAsMatrix(norm(p));
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createNormExpression(expressionLock, this, result, p);
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
            int expressionLock = getProcedureFactory().startExpression(this);
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
    public abstract Matrix applyDropout(double probability, boolean inplace) throws MatrixException;

    /**
     * Clips gradient matrix against threshold.
     *
     * @param threshold threshold.
     * @param inplace if true gradient clipping in done in place otherwise not.
     * @return clipped gradient matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix gradientClip(double threshold, boolean inplace) throws MatrixException {
        Matrix result = this.copy();
        if (hasProcedureFactory()) {
            int expressionLock = getProcedureFactory().startExpression(this);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createGradientClippingExpression(expressionLock, this, result, threshold);
        }
        return result;
    }

    /**
     * Returns softmax of this matrix.
     *
     * @return softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmax() throws MatrixException {
        return softmax(1);
    }

    /**
     * Returns Gumbel softmax of this matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @return Gumbel softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix gumbelSoftmax() throws MatrixException {
        return gumbelSoftmax(1);
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
            int expressionLock = getProcedureFactory().startExpression(this);
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
    protected abstract Matrix applyConvolve(Matrix filter) throws MatrixException;

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
            int expressionLock = getProcedureFactory().startExpression(this);
            Matrix result = applyCrosscorrelate(filter);
            ProcedureFactory.synchronize(this, filter, result);
            getProcedureFactory().createCrosscorrelateExpression(expressionLock, this, filter, result, getStride(), getDilation(), getIsDepthSeparable());
            return result;
        }
    }

    /**
     * Calculates crosscorrelation between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract Matrix applyCrosscorrelate(Matrix filter) throws MatrixException;

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated value of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix winogradConvolve(Matrix filter) throws MatrixException {
        if (!hasProcedureFactory() && !filter.hasProcedureFactory()) return applyWinogradConvolve(filter);
        else {
            ProcedureFactory.synchronize(this, filter);
            int expressionLock = getProcedureFactory().startExpression(this);
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
     */
    protected abstract Matrix applyWinogradConvolve(Matrix filter) throws MatrixException;

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
        if (!hasProcedureFactory() && !filter.hasProcedureFactory()) return applyWinogradConvolve(filter, A, AT, C, CT, G, GT);
        else {
            ProcedureFactory.synchronize(this, filter);
            int expressionLock = getProcedureFactory().startExpression(this);
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
    protected abstract Matrix applyWinogradConvolve(Matrix filter, Matrix A, Matrix AT, Matrix C, Matrix CT, Matrix G, Matrix GT) throws MatrixException;

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
        Matrix result = getNewMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getDepth());
        if (!hasProcedureFactory() && !preprocessedFilter.hasProcedureFactory()) return applyWinogradConvolve(preprocessedFilter, A, AT, C, CT);
        else {
            ProcedureFactory.synchronize(this, preprocessedFilter, result);
            int expressionLock = getProcedureFactory().startExpression(this);
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
    protected abstract Matrix applyWinogradConvolve(Matrix preprocessedFilter, Matrix A, Matrix AT, Matrix C, Matrix CT) throws MatrixException;

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
            int expressionLock = getProcedureFactory().startExpression(this);
            Matrix result = applyMaxPool(maxPos);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createMaxPoolExpression(expressionLock, this, result, getDilation(), getStride(), getFilterRowSize(), getFilterColumnSize());
            return result;
        }
    }

    /**
     * Calculates max pooling operation for this matrix and returns max arguments.
     *
     * @param maxPos maximum positions for each row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract Matrix applyMaxPool(HashMap<Integer, Integer> maxPos) throws MatrixException;

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
            int expressionLock = getProcedureFactory().startExpression(this);
            Matrix result = applyRandomPool(inputPos);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createRandomPoolExpression(expressionLock, this, result, getDilation(), getStride(), getFilterRowSize(), getFilterColumnSize());
            return result;
        }
    }

    /**
     * Calculates random pooling operation for this matrix and returns input positions.
     *
     * @param inputPos input positions for each row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract Matrix applyRandomPool(HashMap<Integer, Integer> inputPos) throws MatrixException;

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
            int expressionLock = getProcedureFactory().startExpression(this);
            Matrix result = applyCyclicPool(inputPos);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createCyclicPoolExpression(expressionLock, this, result, getDilation(), getStride(), getFilterRowSize(), getFilterColumnSize());
            return result;
        }
    }

    /**
     * Calculates cyclic pooling operation for this matrix and returns input positions.
     *
     * @param inputPos input positions for each row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract Matrix applyCyclicPool(HashMap<Integer, Integer> inputPos) throws MatrixException;

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix averagePool() throws MatrixException {
        if (!hasProcedureFactory()) return applyAveragePool();
        else {
            int expressionLock = getProcedureFactory().startExpression(this);
            Matrix result = applyAveragePool();
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createAveragePoolExpression(expressionLock, this, result, getDilation(), getStride(), getFilterRowSize(), getFilterColumnSize());
            return result;
        }
    }

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract Matrix applyAveragePool() throws MatrixException;


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
            int expressionLock = getProcedureFactory().startExpression(this);
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
            int expressionLock = getProcedureFactory().startExpression(this);
            Matrix result = applyUnjoin(unjoinAtRow, unjoinAtColumn, unjoinAtDepth, unjoinRows, unjoinColumns, unjoinDepth);
            ProcedureFactory.synchronize(this, result);
            getProcedureFactory().createUnjoinExpression(expressionLock, this, result, unjoinAtRow, unjoinAtColumn, unjoinAtDepth);
            return result;
        }
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
            int expressionLock = getProcedureFactory().startExpression(this);
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
