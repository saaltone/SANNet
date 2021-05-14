/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.matrix;

import utils.DynamicParamException;
import utils.procedure.ProcedureFactory;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashMap;

/**
 * Abstract class that implements common operations for matrices.<br>
 *
 */
public abstract class AbstractMatrix implements Cloneable, Serializable, Matrix {

    @Serial
    private static final long serialVersionUID = 4372639167186260605L;

    /**
     * Reference to mask of matrix. If null mask is not used.
     *
     */
    private Mask mask;

    /**
     * Procedure factory reference for matrix.
     * Procedure factory records chain of executed matrix operations enabling dynamic construction of procedure and it's gradient.
     *
     */
    private transient ProcedureFactory procedureFactory = null;

    /**
     * If true matrix is normalized otherwise false.
     *
     */
    private boolean normalize = false;

    /**
     * If true matrix is regularized otherwise false.
     *
     */
    private boolean regularize = false;

    /**
     * Name of matrix.
     *
     */
    private String name;

    /**
     * Constructor for matrix.
     *
     */
    protected AbstractMatrix() {
    }

    /**
     * Constructor for matrix.
     *
     * @param name name if matrix.
     */
    protected AbstractMatrix(String name) {
        this.name = name;
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
        if (mask != null) mask.clear();
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
            newMatrix.setInitializer(getInitializer());
            if (getMask() != null) newMatrix.setMask(getMask());
        } catch (CloneNotSupportedException exception) {
            throw new MatrixException("Cloning of matrix failed.");
        }
        return newMatrix;
    }

    /**
     * Creates new matrix with object full copy of this matrix.
     *
     * @return newly created reference matrix.
     * @throws MatrixException throws exception if mask is not set or cloning of matrix fails.
     */
    public Matrix copy() throws MatrixException {
        Matrix newMatrix;
        // Make deep copy of matrix.
        try {
            newMatrix = (Matrix)super.clone();
            newMatrix.setInitializer(getInitializer());
            newMatrix.copyMatrixData(this); // Copy matrix data
            if (getMask() != null) newMatrix.setMask(getMask().copy());
        } catch (CloneNotSupportedException exception) {
            throw new MatrixException("Cloning of matrix failed.");
        }
        return newMatrix;
    }

    /**
     * Slices current matrix.
     *
     * @param startRow start row of slice.
     * @param startColumn start column of slice.
     * @param endRow  end row of slice.
     * @param endColumn  end column of slice.
     * @return sliced matrix.
     * @throws MatrixException throws exception if slicing fails.
     */
    public Matrix slice(int startRow, int startColumn, int endRow, int endColumn) throws MatrixException {
        Matrix referenceMatrix = reference();
        referenceMatrix.sliceAt(startRow, startColumn, endRow, endColumn);
        return referenceMatrix;
    }

    /**
     * Checks if this matrix and other matrix are equal in dimensions (rows x columns).
     *
     * @param other other matrix to be compared against.
     * @return true if matrices are of same size otherwise false.
     */
    public boolean hasEqualSize(Matrix other) {
        return other.getRows() == getRows() || other.getColumns() == getColumns();
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
     * Synchronizes this and other matrix procedure factories.
     *
     * @param other other matrix
     * @throws MatrixException throws exception if his and other matrices have conflicting procedure factories.
     */
    private void synchronizeProcedureFactory(Matrix other) throws MatrixException {
        ProcedureFactory otherProcedureFactory = other.getProcedureFactory();
        if (procedureFactory != otherProcedureFactory) {
            if (procedureFactory == null) setProcedureFactory(otherProcedureFactory);
            else {
                if (otherProcedureFactory == null) other.setProcedureFactory(procedureFactory);
                else throw new MatrixException("This and other matrices have conflicting procedure factories.");
            }
        }
    }

    /**
     * Sets flag if matrix is normalized.
     *
     * @param normalize if true matrix is normalized.
     */
    public void setNormalize(boolean normalize) {
        this.normalize = normalize;
    }

    /**
     * Returns flag if matrix is normalized.
     *
     * @return if true matrix is normalized.
     */
    public boolean isNormalized() {
        return normalize;
    }

    /**
     * Sets flag if matrix is regularized.
     *
     * @param regularize if true matrix is regularized.
     */
    public void setRegularize(boolean regularize) {
        this.regularize = regularize;
    }

    /**
     * Returns flag if matrix is regularized.
     *
     * @return if true matrix is regularized.
     */
    public boolean isRegularized() {
        return regularize;
    }

    /**
     * Returns placeholder for result matrix.
     *
     * @param other other matrix.
     * @return result matrix placeholder.
     */
    private Matrix getResultMatrix(Matrix other) {
        return !isScalar() ? getNewMatrix() : other.getNewMatrix();
    }

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param matrixUnaryOperation single variable operation defined as lambda operator.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix apply(Matrix.MatrixUnaryOperation matrixUnaryOperation) throws MatrixException {
        return apply(getNewMatrix(), matrixUnaryOperation);
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
        if (!hasProcedureFactory()) apply(result, unaryFunction.getFunction());
        else {
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            applyFunction(result, unaryFunction);
            procedureFactory.createUnaryFunctionExpression(expressionLock, this, result, unaryFunction);
        }
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
        Matrix result = apply(getNewMatrix(), unaryFunction.getFunction());
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
     * @param other matrix which acts as second variable in the operation.
     * @param matrixBinaryOperation two variable operation defined as lambda operator.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public Matrix applyBi(Matrix other, Matrix.MatrixBinaryOperation matrixBinaryOperation) throws MatrixException {
        return applyBi(other, getResultMatrix(other), matrixBinaryOperation);
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
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) applyBi(other, result, binaryFunction.getFunction());
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            applyBi(other, result, binaryFunction.getFunction());
            procedureFactory.createBinaryFunctionExpression(expressionLock, this, other, result, binaryFunction);
        }
    }

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
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) Double::sum);
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) Double::sum);
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
        Matrix other = new DMatrix(constant);
        if (!hasProcedureFactory()) add(other, result);
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
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
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 - value2);
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 - value2);
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
        Matrix other = new DMatrix(constant);
        if (!hasProcedureFactory()) subtract(other, result);
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
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
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 * value2);
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 * value2);
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
        Matrix other = new DMatrix(constant);
        if (!hasProcedureFactory()) multiply(other, result);
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
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
        if (!hasProcedureFactory() && !other.hasProcedureFactory()) applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value2 != 0 ? value1 / value2 : Double.POSITIVE_INFINITY);
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 / value2);
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
        Matrix other = new DMatrix(constant);
        if (!hasProcedureFactory()) divide(other, result);
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
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
     * Raises this matrix element wise to the power of value power.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param power power value to which this elements is to be raised.
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
     * @param power power value to which this elements is to be raised.
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
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
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
        return dot(other, new DMatrix(getRows(), other.getColumns()));
    }

    /**
     * Returns constant as matrix.
     *
     * @param constant constant value.
     * @return constant matrix.
     */
    public Matrix constantAsMatrix(double constant) {
        Matrix constantMatrix = new DMatrix(constant);
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
            double expressionLock = procedureFactory.startExpression(this);
            Matrix result = constantAsMatrix(sum());
            result.setProcedureFactory(procedureFactory);
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
            double expressionLock = procedureFactory.startExpression(this);
            Matrix result = constantAsMatrix(mean());
            result.setProcedureFactory(procedureFactory);
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
            double expressionLock = procedureFactory.startExpression(this);
            Matrix result = constantAsMatrix(variance());
            result.setProcedureFactory(procedureFactory);
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
            double expressionLock = procedureFactory.startExpression(this);
            Matrix result = constantAsMatrix(standardDeviation());
            result.setProcedureFactory(procedureFactory);
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
            double expressionLock = procedureFactory.startExpression(this);
            Matrix result = constantAsMatrix(norm(p));
            result.setProcedureFactory(procedureFactory);
            procedureFactory.createNormExpression(expressionLock, this, result, p);
            return result;
        }
    }

    /**
     * Calculates exponential moving average.
     *
     * @param currentAverage current average value
     * @param beta degree of weighting decrease for exponential moving average.
     * @return updated average with new average value included.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix exponentialMovingAverage(Matrix currentAverage, double beta) throws MatrixException {
        return currentAverage == null ? this : currentAverage.multiply(beta).add(multiply(1 - beta));
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
        return other.apply(other, (Matrix.MatrixUnaryOperation & Serializable) (value) -> (value - minimum) / delta * (newMaximum - newMinimum) + newMinimum);
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
     * Returns softmax of this matrix.
     *
     * @return softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmax() throws MatrixException {
        return softmax(new DMatrix(getRows(), getColumns()));
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
        return gumbelSoftmax(new DMatrix(getRows(), getColumns()), 1);
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
        return gumbelSoftmax(new DMatrix(getRows(), getColumns()), gumbelSoftmaxTau);
    }

    /**
     * Returns softmax gradient of this matrix.
     *
     * @return softmax gradient of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmaxGrad() throws MatrixException {
        return softmaxGrad(new DMatrix(getRows(), getRows()));
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated result of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix convolve(Matrix filter) throws MatrixException {
        Matrix result = new DMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1);
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
        Matrix result = new DMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1);
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
            synchronizeProcedureFactory(filter);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            applyConvolve(filter, result);
            procedureFactory.createConvolveExpression(expressionLock, this, filter, result, getStride(), getDilation(), getFilterRowSize(), getFilterColumnSize());
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
            synchronizeProcedureFactory(filter);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            applyCrosscorrelate(filter, result);
            procedureFactory.createCrosscorrelateExpression(expressionLock, this, filter, result, getStride(), getDilation(), getFilterRowSize(), getFilterColumnSize());
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
        Matrix result = new DMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1);
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
            synchronizeProcedureFactory(filter);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            applyWinogradConvolve(filter, result);
            procedureFactory.createWinogradConvolveExpression(expressionLock, this, filter, result, getStride(), getDilation(), getFilterRowSize(), getFilterColumnSize());
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
        Matrix result = new DMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1);
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
            synchronizeProcedureFactory(filter);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            applyWinogradConvolve(filter, result, A, AT, C, CT, G, GT);
            procedureFactory.createWinogradConvolveExpression(expressionLock, this, filter, result, getStride(), getDilation(), getFilterRowSize(), getFilterColumnSize());
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
        Matrix result = new DMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1);
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
            synchronizeProcedureFactory(preprocessedFilter);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            applyWinogradConvolve(preprocessedFilter, result, A, AT, C, CT);
            procedureFactory.createWinogradConvolveExpression(expressionLock, this, preprocessedFilter, result, getStride(), getDilation(), getFilterRowSize(), getFilterColumnSize());
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
        Matrix inputGradient = new DMatrix(getRows() + getFilterRowSize() - 1, getColumns() + getFilterColumnSize() - 1);
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
        Matrix inputGradient = new DMatrix(getRows() + getFilterRowSize() - 1, getColumns() + getFilterColumnSize() - 1);
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
        Matrix filterGradient = new DMatrix(getFilterRowSize(), getFilterColumnSize());
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
        Matrix filterGradient = new DMatrix(getFilterRowSize(), getFilterColumnSize());
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
        Matrix result = new DMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1);
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
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
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
        Matrix inputGradient = new DMatrix(getRows() + getFilterRowSize() - 1, getColumns() + getFilterColumnSize() - 1);
        maxPoolGradient(inputGradient, maxPos);
        return inputGradient;
    }

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix averagePool() throws MatrixException {
        Matrix result = new DMatrix(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1);
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
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
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
        Matrix inputGradient = new DMatrix(getRows() + getFilterRowSize() - 1, getColumns() + getFilterColumnSize() - 1);
        averagePoolGradient(inputGradient);
        return inputGradient;
    }

    /**
     * Transposes matrix.
     *
     * @return new matrix but as transposed with flipped rows and columns.
     */
    public Matrix transpose() {
        Matrix transposedMatrix = getNewMatrix(true);
        for (int row = 0; row < getRows(); row++) {
            for (int column = 0; column < getColumns(); column++) {
                transposedMatrix.setValue(column, row, getValue(row, column));
            }
        }
        if (getMask() != null) {
            transposedMatrix.setMask();
            Mask transposedMask = transposedMatrix.getMask();
            for (int row = 0; row < getRows(); row++) {
                for (int column = 0; column < getColumns(); column++) {
                    if (hasMaskAt(row, column)) transposedMask.setMask(column, row, true);
                }
            }
        }
        return transposedMatrix;
    }

    /**
     * Concatenates this and other matrix vertically.
     *
     * @param other matrix to be concatenated to the end of this matrix vertically.
     * @throws MatrixException throws exception if column dimensions of this and other matrix are not matching.
     */
    public void concatenateVertical(Matrix other) throws MatrixException {
        if (getColumns() != other.getColumns()) throw new MatrixException("Merge Vertical: Incompatible matrix sizes: " + getRows() + "x" + getColumns() + " by " + other.getRows() + "x" + other.getColumns());
        Matrix newMatrix = ((this instanceof SMatrix) && (other instanceof SMatrix)) ? new SMatrix (getRows() + other.getRows(), getColumns()) : new DMatrix (getRows() + other.getRows(), getColumns());
        for (int row = 0; row < getRows(); row++) {
            for (int column = 0; column < getColumns(); column++) {
                newMatrix.setValue(row, column, getValue(row, column));
            }
        }
        for (int row = 0; row < other.getRows(); row++) {
            for (int column = 0; column < other.getColumns(); column++) {
                newMatrix.setValue(getRows() + row, column, other.getValue(row, column));
            }
        }
        copyMatrixData(newMatrix);
    }

    /**
     * Concatenates this and other matrix horizontally.
     *
     * @param other matrix to be concatenated to the end of this matrix horizontally.
     * @throws MatrixException throws exception if row dimensions of this and other matrix are not matching.
     */
    public void concatenateHorizontal(Matrix other) throws MatrixException {
        if (getRows() != other.getRows()) throw new MatrixException("Merge Horizontal: Incompatible matrix sizes: " + getRows() + "x" + getColumns() + " by " + other.getRows() + "x" + other.getColumns());
        Matrix newMatrix = ((this instanceof SMatrix) && (other instanceof SMatrix)) ? new SMatrix(getRows(), getColumns() + other.getColumns()) : new DMatrix(getRows(), getColumns() + other.getColumns());
        for (int row = 0; row < getRows(); row++) {
            for (int column = 0; column < getColumns(); column++) {
                newMatrix.setValue(row, column, getValue(row, column));
            }
        }
        for (int row = 0; row < other.getRows(); row++) {
            for (int column = 0; column < other.getColumns(); column++) {
                newMatrix.setValue(row, getColumns() + column, other.getValue(row, column));
            }
        }
        copyMatrixData(newMatrix);
    }

    /**
     * Prints matrix in row and column format.
     *
     */
    public void print() {
        for (int row = 0; row < getRows(); row++) {
            System.out.print("[");
            for (int column = 0; column < getColumns(); column++) {
                System.out.print(getValue(row, column));
                if (column < getColumns() - 1) System.out.print(" ");
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
     * Checks if matrix has mask at specific position.
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
