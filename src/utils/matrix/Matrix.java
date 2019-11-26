/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package utils.matrix;

import core.normalization.Normalization;
import utils.procedure.ProcedureFactory;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Random;

/**
 * Class that implements matrix with extensive set of matrix operations and masking for matrix.<br>
 * Abstract matrix class to be extended by class providing matrix data structure implementation.<br>
 * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
 * Supports automated calculation of gradients by recording operations into procedure factory.<br>
 *
 */
public abstract class Matrix implements Cloneable, Serializable {

    private static final long serialVersionUID = 578590231908806892L;

    /**
     * Defines interface to be used as part of lambda function to initialize Matrix.
     */
    public interface Initializer {
        /**
         * Returns value to be used for initialization.
         *
         * @param row row of matrix if relevant for initialization.
         * @param col col of matrix if relevant for initialization.
         * @return value to be used for initialization.
         */
        double value(int row, int col);
    }

    /**
     * Defines interface to be used as part of lambda function to execute two argument matrix operation.
     */
    public interface MatrixBinaryOperation {
        /**
         * Defines operation to be executed with two parameters.
         *
         * @param value1 value for first parameter.
         * @param value2 value for second parameter.
         * @return value returned by the operation.
         */
        double execute(double value1, double value2);
    }

    /**
     * Defines interface to be used as part of lambda function to execute single argument matrix operation.
     */
    public interface MatrixUnaryOperation {
        /**
         * Defines operation to be executed with single parameter.
         *
         * @param value1 value for parameter.
         * @return value returned by the operation.
         */
        double execute(double value1);
    }

    /**
     * Defines if matrix is transposed (true) or not (false).
     */
    boolean t;

    /**
     * Initializer variable.
     */
    private Initializer initializer;

    /**
     * Initialization type
     *
     */
    private Init init = Init.ZERO;

    /**
     * Reference to mask of matrix.
     */
    private Mask mask;

    /**
     * Scaling constant applied in all matrix operations meaning scalingConstant * operation.
     */
    private double scalingConstant = 1;

    /**
     * If true matrix is treated as constant (1x1) matrix otherwise as normal matrix.
     *
     */
    private boolean isConstant = false;

    /**
     * Stride size for convolutional operator. Default 1.
     *
     */
    private int stride = 1;

    /**
     * Pool size. Default 2.
     *
     */
    private int poolSize = 2;

    /**
     * Autogradient for matrix.
     *
     */
    private transient ProcedureFactory procedureFactory = null;

    /**
     * Procedure callback for the matrix.
     *
     */
    private HashSet<Normalization> normalizers = null;

    /**
     * Random function for matrix class.
     */
    private final Random random = new Random();

    /**
     * Function used to reinitialize matrix and it's mask.
     */
    public void reset() {
        resetMatrix();
        if (mask != null) mask.clear();
    }

    /**
     * Abstract matrix reset function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     */
    protected abstract void resetMatrix();

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
     * Returns value from normal distribution defined by standard deviation (stdev).
     *
     * @param stdev standard deviation of the distribution.
     * @return random value drawn from the distribution.
     */
    private double normal(double stdev) {
        return random.nextGaussian() * stdev;
    }

    /**
     * Default initialization without parameter.
     *
     * @param init type of initialization defined in class Init.
     * @throws MatrixException throws exception if initialization fails.
     */
    public void init(Init init) throws MatrixException {
        init(init, 0, 0);
    }

    /**
     * Default initialization with parameter.
     *
     * @param init type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void init(Init init, int inputs, int outputs) throws MatrixException {
        this.init = init;
        switch (init) {
            case ZERO:
                initializer = (Initializer & Serializable) (row, col) -> 0;
                break;
            case ONE:
                initializer = (Initializer & Serializable) (row, col) -> 1;
                initialize(initializer);
                break;
            case CONSTANT:
                if (getRows() != 1 && getCols() != 1) throw new MatrixException("Cannot initialized as constant because matrix dimension is not 1x1");
                isConstant = true;
                initializer = (Initializer & Serializable) (row, col) -> 0;
                break;
            case RANDOM:
                initializer = (Initializer & Serializable) (row, col) -> random.nextDouble();
                initialize(initializer);
                break;
            case IDENTITY:
                initializer = (Initializer & Serializable) (row, col) -> (row == col) ? 1 : 0;
                initialize(initializer);
                break;
            case NORMAL_XAVIER:
                initializer = (Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / (double)(getRows() + getCols())));
                initialize(initializer);
                break;
            case UNIFORM_XAVIER:
                initializer = (Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(getRows() + getCols())));
                initialize(initializer);
                break;
            case NORMAL_HE:
                initializer = (Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / ((double)getRows())));
                initialize(initializer);
                break;
            case UNIFORM_HE:
                initializer = (Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(getRows())));
                initialize(initializer);
                break;
            case NORMAL_LECUN:
                initializer = (Initializer & Serializable) (row, col) -> normal(Math.sqrt(1 / (double)(getRows())));
                initialize(initializer);
                break;
            case UNIFORM_LECUN:
                initializer = (Initializer & Serializable) (row, col) -> uniform(Math.sqrt(3 / (double)(getRows())));
                initialize(initializer);
                break;
            case NORMAL_XAVIER_CONV:
                initializer = (Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / (double)(outputs + inputs)));
                initialize(initializer);
                break;
            case UNIFORM_XAVIER_CONV:
                initializer = (Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(outputs + inputs)));
                initialize(initializer);
                break;
            case NORMAL_HE_CONV:
                initializer = (Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / (double)(outputs)));
                initialize(initializer);
                break;
            case UNIFORM_HE_CONV:
                initializer = (Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(outputs)));
                initialize(initializer);
                break;
            case NORMAL_LECUN_CONV:
                initializer = (Initializer & Serializable) (row, col) -> normal(Math.sqrt(1 / (double)(outputs)));
                initialize(initializer);
                break;
            case UNIFORM_LECUN_CONV:
                initializer = (Initializer & Serializable) (row, col) -> uniform(Math.sqrt(3 / (double)(outputs)));
                initialize(initializer);
                break;
            default:
                break;
        }
    }

    public boolean isConstant() {
        return isConstant;
    }

    /**
     * Returns matrix init type.
     *
     * @return current matrix initializer instance.
     */
    public Init getInitType() {
        return init;
    }

    /**
     * Returns matrix initializer.
     *
     * @return current matrix initializer instance.
     */
    public Initializer getInitializer() {
        return initializer;
    }

    /**
     * Initializes matrix with given initializer operation.
     *
     * @param operation initializer operation.
     */
    public void initialize(Initializer operation) {
        for (int row = 0; row < getRows(); row++) {
            for (int col = 0; col < getCols(); col++) {
                setValue(row, col, operation.value(row, col));
            }
        }
    }

    /**
     * Initializes matrix with given value.
     *
     * @param value initialization value.
     */
    public void initializeToValue(double value) {
        for (int row = 0; row < getRows(); row++) {
            for (int col = 0; col < getCols(); col++) {
                setValue(row, col, value);
            }
        }
    }

    /**
     * Matrix internal function used to set value of specific row and column.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @param row row of value to be set.
     * @param col column of value to be set.
     * @param value new value to be set.
     */
    public abstract void setValue(int row, int col, double value);

    /**
     * Matrix internal function used to get value of specific row and column.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @param row row of value to be returned.
     * @param col column of value to be returned.
     * @return value of row and column.
     */
    public abstract double getValue(int row, int col);

    /**
     * Returns size (rows * columns) of matrix.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @return size of matrix.
     */
    public abstract int getSize();

    /**
     * Returns number of rows in matrix.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @return number of rows in matrix.
     */
    public abstract int getRows();

    /**
     * Returns number of columns in matrix.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @return number of columns in matrix.
     */
    public abstract int getCols();

    /**
     *
     * Matrix function used to add value of specific row and column.
     *
     * @param row row of value to be added.
     * @param col column of value to be added.
     * @param value to be added.
     */
    public void addValue(int row, int col, double value) {
        setValue(row, col, getValue(row, col) + value);
    }

    /**
     *
     * Matrix function used to decrease value of specific row and column.
     *
     * @param row row of value to be decreased.
     * @param col column of value to be decreased.
     * @param value to be decreased.
     */
    public void decValue(int row, int col, double value) {
        setValue(row, col, getValue(row, col) - value);
    }

    /**
     *
     * Matrix function used to multiply value of specific row and column.
     *
     * @param row row of value to be multiplied.
     * @param col column of value to be multiplied.
     * @param value to be multiplied.
     */
    public void mulValue(int row, int col, double value) {
        setValue(row, col, getValue(row, col) * value);
    }

    /**
     *
     * Matrix function used to divide value of specific row and column.
     *
     * @param row row of value to be divided.
     * @param col column of value to be divided.
     * @param value to be divided.
     */
    public void divValue(int row, int col, double value) {
        setValue(row, col, getValue(row, col) / value);
    }

    /**
     * Returns new matrix of dimensions rows x columns.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @param rows amount of rows for new matrix.
     * @param cols amount of columns for new matrix.
     * @return new matrix of dimensions rows x columns.
     */
    protected abstract Matrix getNewMatrix(int rows, int cols);

    /**
     * Copies new matrix inside this matrix with dimensions rows x columns.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @param newMatrix new matrix to be copied inside this matrix.
     */
    protected abstract void setAsMatrix(Matrix newMatrix);

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
            newMatrix.initializer = initializer;
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
        // Make shallow copy of matrix leaving references internal objects which are shared.
        try {
            newMatrix = (Matrix)super.clone();
            newMatrix.initializer = initializer;
            // Copy matrix data
            newMatrix.setAsMatrix(this);
            if (getMask() != null) newMatrix.setMask(getMask().copy());
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
        return other.getRows() == getRows() || other.getCols() == getCols();
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
     * Returns and propagates current procedure factory.
     *
     * @param result result matrix
     * @return returns current procedure factory
     */
    private ProcedureFactory getProcedureFactory(Matrix result) {
        result.setProcedureFactory(procedureFactory);
        return procedureFactory;
    }

    /**
     * Returns and propagates current procedure factory.
     *
     * @param other other matrix
     * @param result result matrix
     * @return returns current procedure factory
     */
    private ProcedureFactory getProcedureFactory(Matrix other, Matrix result) throws MatrixException {
        ProcedureFactory otherProcedureFactory = other.getProcedureFactory();
        if (procedureFactory == null && otherProcedureFactory == null) return null;
        if (procedureFactory != null && otherProcedureFactory != null && procedureFactory != otherProcedureFactory) throw new MatrixException("This and other matrices have conflicting procedure factories.");
        ProcedureFactory currentProcedureFactory = procedureFactory != null ? procedureFactory : otherProcedureFactory;
        result.setProcedureFactory(currentProcedureFactory);
        return currentProcedureFactory;
    }

    /**
     * Sets procedure callback for the matrix.
     *
     * @param normalizers reference to procedure callback.
     */
    public void setNormalization(HashSet<Normalization> normalizers) {
        this.normalizers = normalizers;
    }

    /**
     * Returns procedure callback.
     *
     * @return procedure callback.
     */
    public HashSet<Normalization> getNormalization() {
        return normalizers;
    }

    /**
     * Makes current matrix data equal to other matrix data.
     *
     * @param other other matrix to be copied as data of this matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void setEqualTo(Matrix other) throws MatrixException {
        if (other.getRows() != getRows() || other.getCols() != getCols()) {
            throw new MatrixException("Incompatible target matrix size: " + other.getRows() + "x" + other.getCols());
        }
        for (int row = 0; row < other.getRows(); row++) {
            for (int col = 0; col < other.getCols(); col++) {
                setValue(row, col, other.getValue(row, col));
                if (getMask() !=null) getMask().setMask(row, col, getMask(other, row, col));
            }
        }
    }

    /**
     * Checks if data of other matrix is equal to data of this matrix
     *
     * @param other matrix to be compared.
     * @return true is data of this and other matrix are equal otherwise false.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public boolean equals(Matrix other) throws MatrixException {
        if (other.getRows() != getRows() || other.getCols() != getCols()) {
            throw new MatrixException("Incompatible target matrix size: " + other.getRows() + "x" + other.getCols());
        }
        for (int row = 0; row < other.getRows(); row++) {
            for (int col = 0; col < other.getCols(); col++) {
                if (getValue(row, col) != other.getValue(row, col)) return false;
            }
        }
        return true;
    }

    /**
     * Applies single variable operation to this matrix and stores operation result into result matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param result matrix which stores operation result.
     * @param operation single variable operation defined as lambda operator.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and result matrix are not of equal dimensions.
     */
    public Matrix apply(Matrix result, MatrixUnaryOperation operation) throws MatrixException {
        if (result.getRows() != getRows() || result.getCols() != getCols()) {
            throw new MatrixException("Incompatible result matrix sizes: " + result.getRows() + "x" + result.getCols());
        }
        if (getMask() == null) {
            for (int row = 0; row < getRows(); row++) {
                for (int col = 0; col < getCols(); col++) {
                    result.setValue(row, col, scalingConstant * operation.execute(getValue(row, col)));
                }
            }
        }
        else {
            for (int row = 0; row < getRows(); row++) {
                if (getRowMask(this, row)) {
                    for (int col = 0; col < getCols(); col++) {
                        if (!getMask(this, row, col) && getColMask(this, col)) {
                            result.setValue(row, col, scalingConstant * operation.execute(getValue(row, col)));
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * Applies single variable operation to this matrix and return operation result.<br>
     * Example of operation can be applying square root operation to this matrix or
     * multiplying current matrix with constant number.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param operation single variable operation defined as lambda operator.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix apply(MatrixUnaryOperation operation) throws MatrixException {
        return apply(getNewMatrix(getRows(), getCols()), operation);
    }

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param unaryFunctionType unaryFunction type to be applied.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(Matrix result, UnaryFunctionType unaryFunctionType) throws MatrixException {
        UnaryFunction unaryFunction = new UnaryFunction(unaryFunctionType);
        apply(result, unaryFunction.getFunction());
        ProcedureFactory currentProcedureFactory = getProcedureFactory(result);
        if (currentProcedureFactory != null) currentProcedureFactory.createUnaryFunctionExpression(this, result, unaryFunction, normalizers);
    }

    /**
     * Applies unaryFunction to this matrix and return operation result.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param unaryFunctionType unaryFunction type to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix apply(UnaryFunctionType unaryFunctionType) throws MatrixException {
        UnaryFunction unaryFunction = new UnaryFunction(unaryFunctionType);
        Matrix result = apply(getNewMatrix(getRows(), getCols()), unaryFunction.getFunction());
        ProcedureFactory currentProcedureFactory = getProcedureFactory(result);
        if (currentProcedureFactory != null) currentProcedureFactory.createUnaryFunctionExpression(this, result, unaryFunction, normalizers);
        return result;
    }

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param unaryFunction unaryFunction to be applied.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(Matrix result, UnaryFunction unaryFunction) throws MatrixException {
        apply(result, unaryFunction.getFunction());
        ProcedureFactory currentProcedureFactory = getProcedureFactory(result);
        if (currentProcedureFactory != null) currentProcedureFactory.createUnaryFunctionExpression(this, result, unaryFunction, normalizers);
    }

    /**
     * Applies unaryFunction to this matrix and return operation result.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param unaryFunction unaryFunction to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix apply(UnaryFunction unaryFunction) throws MatrixException {
        Matrix result = apply(getNewMatrix(getRows(), getCols()), unaryFunction.getFunction());
        ProcedureFactory currentProcedureFactory = getProcedureFactory(result);
        if (currentProcedureFactory != null) currentProcedureFactory.createUnaryFunctionExpression(this, result, unaryFunction, normalizers);
        return result;
    }

    /**
     * Applies two variable operation to this matrix and other matrix and stores operation result into result matrix.<br>
     * Example of operation can be substraction of other matrix from this matrix or
     * multiplying current matrix with other matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the opration.
     * @param result matrix which stores operation result.
     * @param operation two variable operation defined as lambda operator.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     */
    public Matrix applyBi(Matrix other, Matrix result, Matrix.MatrixBinaryOperation operation) throws MatrixException {
        if (getRows() != other.getRows() || getCols() != other.getCols()) {
            throw new MatrixException("Incompatible matrix sizes: " + getRows() + "x" + getCols() + " by " + other.getRows() + "x" + other.getCols());
        }
        if (getRows() != result.getRows() || getCols() != result.getCols()) {
            throw new MatrixException("Incompatible result matrix sizes: " + result.getRows() + "x" + result.getCols());
        }
        if (getMask() == null && other.getMask() == null) {
            for (int row = 0; row < getRows(); row++) {
                for (int col = 0; col < getCols(); col++) {
                    result.setValue(row, col, scalingConstant * operation.execute(getValue(row, col), other.getValue(row, col)));
                }
            }
        }
        else {
            for (int row = 0; row < getRows(); row++) {
                if (getRowMask(this, row) && getRowMask(other, row)) {
                    for (int col = 0; col < getCols(); col++) {
                        if (!getMask(this, row, col) && getColMask(this, col) && !getMask(other, row, col) && getColMask(other, col)) {
                            result.setValue(row, col, scalingConstant * operation.execute(getValue(row, col), other.getValue(row, col)));
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * Applies two variable operation to this matrix and other matrix and stores operation result into result matrix.<br>
     * Example of operation can be substraction of other matrix from this matrix or
     * multiplying current matrix with other matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the opration.
     * @param operation two variable operation defined as lambda operator.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public Matrix applyBi(Matrix other, Matrix.MatrixBinaryOperation operation) throws MatrixException {
        return applyBi(other, getNewMatrix(getRows(), getCols()), operation);
    }

    /**
     * Applies binaryFunction to this matrix.<br>
     * Example of operation can be applying power operation to this and other matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param other other matrix
     * @param result result matrix.
     * @param binaryFunctionType binaryFunction type to be applied.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyBi(Matrix other, Matrix result, BinaryFunctionType binaryFunctionType) throws MatrixException {
        BinaryFunction binaryFunction = new BinaryFunction(binaryFunctionType);
        applyBi(other, result, binaryFunction.getFunction());
        ProcedureFactory currentProcedureFactory = getProcedureFactory(other, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createBinaryFunctionExpression(this, other, result, binaryFunction, normalizers);
    }

    /**
     * Applies binaryFunction to this matrix and return operation result.<br>
     * Example of operation can be applying power operation to this and other matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param other other matrix
     * @param binaryFunctionType binaryFunction type to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix applyBi(Matrix other, BinaryFunctionType binaryFunctionType) throws MatrixException {
        BinaryFunction binaryFunction = new BinaryFunction(binaryFunctionType);
        Matrix result = applyBi(other, getNewMatrix(getRows(), getCols()), binaryFunction.getFunction());
        ProcedureFactory currentProcedureFactory = getProcedureFactory(other, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createBinaryFunctionExpression(this, other, result, binaryFunction, normalizers);
        return result;
    }

    /**
     * Applies binaryFunction to this matrix.<br>
     * Example of operation can be applying power operation to this and other matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param other other matrix
     * @param result result matrix.
     * @param binaryFunction binaryFunction to be applied.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyBi(Matrix other, Matrix result, BinaryFunction binaryFunction) throws MatrixException {
        applyBi(other, result, binaryFunction.getFunction());
        ProcedureFactory currentProcedureFactory = getProcedureFactory(other, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createBinaryFunctionExpression(this, other, result, binaryFunction, normalizers);
    }

    /**
     * Applies binaryFunction to this matrix and return operation result.<br>
     * Example of operation can be applying power operation to this and other matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param other other matrix
     * @param binaryFunction binaryFunction to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix applyBi(Matrix other, BinaryFunction binaryFunction) throws MatrixException {
        Matrix result = applyBi(other, getNewMatrix(getRows(), getCols()), binaryFunction.getFunction());
        ProcedureFactory currentProcedureFactory = getProcedureFactory(other, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createBinaryFunctionExpression(this, other, result, binaryFunction, normalizers);
        return result;
    }

    /**
     * Adds other matrix to this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the opration.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void add(Matrix other, Matrix result) throws MatrixException {
        if (isConstant() == other.isConstant()) applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) Double::sum);
        else {
            if (isConstant()) other.apply (result, (Matrix.MatrixUnaryOperation & Serializable) (value) ->  getValue(0,0) + value);
            else apply (result, (Matrix.MatrixUnaryOperation & Serializable) (value) -> value + other.getValue(0,0));
        }
        ProcedureFactory currentProcedureFactory = getProcedureFactory(other, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createAddExpression(this, other, result, normalizers);
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
        Matrix result;
        if (isConstant() == other.isConstant()) result = applyBi (other, (Matrix.MatrixBinaryOperation & Serializable) Double::sum);
        else {
            if (isConstant()) result = other.apply ((Matrix.MatrixUnaryOperation & Serializable) (value) -> getValue(0,0) +  value);
            else result = apply ((Matrix.MatrixUnaryOperation & Serializable) (value) -> value + other.getValue(0,0));
        }
        ProcedureFactory currentProcedureFactory = getProcedureFactory(other, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createAddExpression(this, other, result, normalizers);
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
        Matrix constantMatrix = new DMatrix(getRows(), getCols());
        constantMatrix.initializeToValue(constant);
        add(constantMatrix, result);
        ProcedureFactory currentProcedureFactory = getProcedureFactory(constantMatrix, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createAddExpression(this, constantMatrix, result, normalizers);
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
        Matrix constantMatrix = new DMatrix(getRows(), getCols());
        constantMatrix.initializeToValue(constant);
        Matrix result = add(constantMatrix);
        ProcedureFactory currentProcedureFactory = getProcedureFactory(constantMatrix, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createAddExpression(this, constantMatrix, result, normalizers);
        return result;
    }

    /**
     * Subtracts other matrix from this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the opration.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void subtract(Matrix other, Matrix result) throws MatrixException {
        if (isConstant() == other.isConstant()) applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 - value2);
        else {
            if (isConstant()) other.apply (result, (Matrix.MatrixUnaryOperation & Serializable) (value) ->  getValue(0,0) - value);
            else apply (result, (Matrix.MatrixUnaryOperation & Serializable) (value) -> value - other.getValue(0,0));
        }
        ProcedureFactory currentProcedureFactory = getProcedureFactory(other, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createSubtractExpression(this, other, result, normalizers);
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
        Matrix result;
        if (isConstant() == other.isConstant()) result = applyBi (other, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 - value2);
        else {
            if (isConstant()) result = other.apply ((Matrix.MatrixUnaryOperation & Serializable) (value) ->  getValue(0,0) - value);
            else result = apply ((Matrix.MatrixUnaryOperation & Serializable) (value) -> value - other.getValue(0,0));
        }
        ProcedureFactory currentProcedureFactory = getProcedureFactory(other, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createSubtractExpression(this, other, result, normalizers);
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
        Matrix constantMatrix = new DMatrix(getRows(), getCols());
        constantMatrix.initializeToValue(constant);
        subtract(constantMatrix, result);
        ProcedureFactory currentProcedureFactory = getProcedureFactory(constantMatrix, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createSubtractExpression(this, constantMatrix, result, normalizers);
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
        Matrix constantMatrix = new DMatrix(getRows(), getCols());
        constantMatrix.initializeToValue(constant);
        Matrix result = subtract(constantMatrix);
        ProcedureFactory currentProcedureFactory = getProcedureFactory(constantMatrix, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createSubtractExpression(this, constantMatrix, result, normalizers);
        return result;
    }

    /**
     * Multiplies other matrix element wise with this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the opration.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void multiply(Matrix other, Matrix result) throws MatrixException {
        if (isConstant() == other.isConstant()) applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 * value2);
        else {
            if (isConstant()) other.apply (result, (Matrix.MatrixUnaryOperation & Serializable) (value) ->  getValue(0,0) * value);
            else apply (result, (Matrix.MatrixUnaryOperation & Serializable) (value) -> value * other.getValue(0,0));
        }
        ProcedureFactory currentProcedureFactory = getProcedureFactory(other, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createMultiplyExpression(this, other, result, normalizers);
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
        Matrix result;
        if (isConstant() == other.isConstant()) result = applyBi (other, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 * value2);
        else {
            if (isConstant()) result = other.apply ((Matrix.MatrixUnaryOperation & Serializable) (value) ->  getValue(0,0) * value);
            else result = apply ((Matrix.MatrixUnaryOperation & Serializable) (value) -> value * other.getValue(0,0));
        }
        ProcedureFactory currentProcedureFactory = getProcedureFactory(other, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createMultiplyExpression(this, other, result, normalizers);
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
        Matrix constantMatrix = new DMatrix(getRows(), getCols());
        constantMatrix.initializeToValue(constant);
        multiply(constantMatrix, result);
        ProcedureFactory currentProcedureFactory = getProcedureFactory(constantMatrix, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createMultiplyExpression(this, constantMatrix, result, normalizers);
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
        Matrix constantMatrix = new DMatrix(getRows(), getCols());
        constantMatrix.initializeToValue(constant);
        Matrix result = multiply(constantMatrix);
        ProcedureFactory currentProcedureFactory = getProcedureFactory(constantMatrix, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createMultiplyExpression(this, constantMatrix, result, normalizers);
        return result;
    }

    /**
     * Divides this matrix element wise with other matrix.<br>
     * In case any element value of other matrix is zero result is treated as Double MAX value to avoid infinity condition.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     */
    public void divide(Matrix other, Matrix result) throws MatrixException {
        if (isConstant() == other.isConstant()) applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value2 != 0 ? value1 / value2 : Double.MAX_VALUE);
        else {
            if (isConstant()) other.apply (result, (Matrix.MatrixUnaryOperation & Serializable) (value) ->  value != 0 ? getValue(0,0) / value : Double.MAX_VALUE);
            else apply (result, (Matrix.MatrixUnaryOperation & Serializable) (value) -> other.getValue(0,0) != 0 ? value / other.getValue(0,0) : Double.MAX_VALUE);
        }
        ProcedureFactory currentProcedureFactory = getProcedureFactory(other, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createDivideExpression(this, other, result, normalizers);
    }

    /**
     * Divides this matrix element wise with other matrix.<br>
     * In case any element value of other matrix is zero result is treated as Double MAX value to avoid infinity condition.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public Matrix divide(Matrix other) throws MatrixException {
        Matrix result;
        if (isConstant() == other.isConstant()) result = applyBi (other, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value2 != 0 ? value1 / value2 : Double.MAX_VALUE);
        else {
            if (isConstant()) result = other.apply ((Matrix.MatrixUnaryOperation & Serializable) (value) ->  value != 0 ? getValue(0,0) / value : Double.MAX_VALUE);
            else result = apply ((Matrix.MatrixUnaryOperation & Serializable) (value) -> other.getValue(0,0) != 0 ? value / other.getValue(0,0) : Double.MAX_VALUE);
        }
        ProcedureFactory currentProcedureFactory = getProcedureFactory(other, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createDivideExpression(this, other, result, normalizers);
        return result;
    }

    /**
     * Divides this matrix element wise with constant.<br>
     * In case constant is zero result is treated as Double MAX value to avoid infinity condition.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant constant used as divider value.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and result matrix are not of equal dimensions.
     */
    public void divide(double constant, Matrix result) throws MatrixException {
        Matrix constantMatrix = new DMatrix(getRows(), getCols());
        constantMatrix.initializeToValue(constant);
        divide(constantMatrix, result);
        ProcedureFactory currentProcedureFactory = getProcedureFactory(constantMatrix, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createDivideExpression(this, constantMatrix, result, normalizers);
    }

    /**
     * Divides this matrix element wise with constant.<br>
     * In case constant is zero result is treated as Double MAX value to avoid infinity condition.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant constant used as divider value.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix divide(double constant) throws MatrixException {
        Matrix constantMatrix = new DMatrix(getRows(), getCols());
        constantMatrix.initializeToValue(constant);
        Matrix result = divide(constantMatrix);
        ProcedureFactory currentProcedureFactory = getProcedureFactory(constantMatrix, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createDivideExpression(this, constantMatrix, result, normalizers);
        return result;
    }

    /**
     * Raises this matrix element wise to the power of value power.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param power power value to which this elements is to be raised.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix power(double power) throws MatrixException {
        return apply ((Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.pow(value, power));
    }

    /**
     * Raises this matrix element wise to the power of value power.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param power power value to which this elements is to be raised.
     * @param result matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public void power(double power, Matrix result) throws MatrixException {
        apply (result, (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.pow(value, power));
    }

    /**
     * Takes element wise max value of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     */
    public void max(Matrix other, Matrix result) throws MatrixException {
        applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) Math::max);
    }

    /**
     * Takes element wise max value of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public Matrix max(Matrix other) throws MatrixException {
        return applyBi (other, (Matrix.MatrixBinaryOperation & Serializable) Math::max);
    }

    /**
     * Takes element wise min value of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     */
    public void min(Matrix other, Matrix result) throws MatrixException {
        applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) Math::min);
    }

    /**
     * Takes element wise min value of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public Matrix min(Matrix other) throws MatrixException {
        return applyBi (other, (Matrix.MatrixBinaryOperation & Serializable) Math::min);
    }

    /**
     * Takes element wise signum over multiplication of this and other matrix.<br>
     * Applies first sign operation to each value and then multiplies them.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     */
    public void sgnmul(Matrix other, Matrix result) throws MatrixException {
        applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> Math.signum(value1) * Math.signum(value2));
    }

    /**
     * Takes element wise signum over multiplication of this and other matrix.<br>
     * Applies first sign operation to each value and then multiplies them.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public Matrix sgnmul(Matrix other) throws MatrixException {
        return applyBi (other, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> Math.signum(value1) * Math.signum(value2));
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
        if (getCols() != other.getRows()) {
            throw new MatrixException("Incompatible matrix sizes: " + getRows() + "x" + getCols() + " by " + other.getRows() + "x" + other.getCols());
        }
        if (getRows() != result.getRows() || other.getCols() != result.getCols()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getCols());
        }
        ProcedureFactory currentProcedureFactory = getProcedureFactory(other, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createDotExpression(this, other, result, normalizers);
        if (getMask() == null && other.getMask() == null) {
            for (int row = 0; row < getRows(); row++) {
                for (int col = 0; col < other.getCols(); col++) {
                    result.setValue(row, col, 0);
                    for (int x = 0; x < getCols(); x++) {
                        result.setValue(row, col, result.getValue(row, col) + scalingConstant * getValue(row, x) * other.getValue(x, col));
                    }
                }
            }
        }
        else {
            for (int row = 0; row < getRows(); row++) {
                if (getRowMask(this, row)) {
                    for (int col = 0; col < other.getCols(); col++) {
                        if (getColMask(other, col)) {
                            for (int x = 0; x < getCols(); x++) {
                                if (!getMask(this, row, x) && !getMask(other, x, col)) {
                                    result.setValue(row, col, result.getValue(row, col) + scalingConstant * getValue(row, x) * other.getValue(x, col));
                                }
                            }
                        }
                    }
                }
            }
        }
        return result;
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
        return dot(other, getNewMatrix(getRows(), other.getCols()));
    }

    /**
     * Takes cumulative sum of single variable operation applied over each element of this matrix.<br>
     * Returns result array which has first element containing cumulative sum and second element number of elements.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param operation single variable operation defined as lambda operator.
     * @return array containing cumulative sum and element count as elements.
     */
    private double[] count(MatrixUnaryOperation operation) {
        double[] result = new double[2];
        result[1] = 0;
        if (getMask() == null) {
            for (int row = 0; row < getRows(); row++) {
                for (int col = 0; col < getCols(); col++) {
                    result[0] += operation.execute(getValue(row, col));
                    result[1]++;
                }
            }
        }
        else {
            for (int row = 0; row < getRows(); row++) {
                if (getRowMask(this, row)) {
                    for (int col = 0; col < getCols(); col++) {
                        if (!getMask(this, row, col) && getColMask(this, col)) {
                            result[0] += operation.execute(getValue(row, col));
                            result[1]++;
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * Takes element wise cumulative sum of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return cumulative sum of this matrix.
     */
    public double sum() {
        MatrixUnaryOperation operation = (Matrix.MatrixUnaryOperation & Serializable) value -> value;
        double[] result = count(operation);
        return result[0];
    }

    /**
     * Takes mean of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return mean of elements of this matrix.
     */
    public double mean() {
        MatrixUnaryOperation operation = (Matrix.MatrixUnaryOperation & Serializable) value -> value;
        double[] result = count(operation);
        return result[1] > 0 ? result[0] / result[1] : 0;
    }

    /**
     * Takes variance of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return variance of elements of this matrix.
     */
    public double var() {
        double mean = sum();
        MatrixUnaryOperation operation = (Matrix.MatrixUnaryOperation & Serializable) value -> Math.pow(value - mean, 2);
        double[] result = count(operation);
        return result[1] > 0 ? result[0] / result[1] : 0;
    }

    /**
     * Takes variance of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return variance of elements of this matrix.
     */
    public double var(double mean) {
        MatrixUnaryOperation operation = (Matrix.MatrixUnaryOperation & Serializable) value -> Math.pow(value - mean, 2);
        double[] result = count(operation);
        return result[1] > 0 ? result[0] / result[1] : 0;
    }

    /**
     * Takes standard deviation of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return standard deviation of elements of this matrix.
     */
    public double std() {
        double mean = sum();
        MatrixUnaryOperation operation = (Matrix.MatrixUnaryOperation & Serializable) value -> Math.pow(value - mean, 2);
        double[] result = count(operation);
        return result[1] > 0 ? Math.sqrt(result[0] / (result[1] - 1)) : 0;
    }

    /**
     * Takes standard deviation of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return standard deviation of elements of this matrix.
     */
    public double std(double mean) {
        MatrixUnaryOperation operation = (Matrix.MatrixUnaryOperation & Serializable) value -> Math.pow(value - mean, 2);
        double[] result = count(operation);
        return result[1] > 0 ? Math.sqrt(result[0] / (result[1] - 1)) : 0;
    }

    /**
     * Takes cumulative p- norm (p is number equal or bigger than 1) of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param p p value for norm.
     * @return cumulative norm value of matrix.
     */
    public double norm(int p) {
        MatrixUnaryOperation operation = (Matrix.MatrixUnaryOperation & Serializable) value -> Math.pow(Math.abs(value), p);
        double[] result = count(operation);
        return result[0];
    }

    /**
     * Normalizes other matrix by removing mean and standard deviation.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix to be normalized.
     * @return normalized matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public static Matrix normalize(Matrix other) throws MatrixException {
        double mean = other.mean();
        double std = other.std();
        return other.apply(other, (Matrix.MatrixUnaryOperation & Serializable) (value) -> (value - mean) / std);
    }

    /**
     * Normalizes (scales) this matrix to new min and max values.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param newMin new minimum value.
     * @param newMax new maximum value.
     * @throws MatrixException not thrown in any situation.
     */
    public void minMax(double newMin, double newMax) throws MatrixException {
        Matrix.minMax(this, newMin, newMax);
    }

    /**
     * Normalizes (scales) other matrix to new min and max values.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other other matrix to be scaled.
     * @param newMin new minimum value.
     * @param newMax new maximum value.
     * @return scaled result matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public static Matrix minMax(Matrix other, double newMin, double newMax) throws MatrixException {
        double min = other.min();
        double max = other.max();
        double delta = max - min != 0 ? max - min : 1;
        return other.apply(other, (Matrix.MatrixUnaryOperation & Serializable) (value) -> (value - min) / delta * (newMax - newMin) + newMin);
    }

    /**
     * Finds minimum or maximum element of matrix and return this value with row and column information.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param min If true finds minimum value with row and column information otherwise maximum value.
     * @param index two dimensional array used to return minimum or maximum value row and column in this order.
     * @return minimum or maximum value found.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public double argMinMax(boolean min, int[] index) throws MatrixException {
        if (index.length != 2) throw new MatrixException("Dimension of index must be 2.");
        double value = min ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        if (getMask() == null) {
            for (int row = 0; row < getRows(); row++) {
                for (int col = 0; col < getCols(); col++) {
                    if (min) {
//                        double curValue = Math.min(value, getValue(row, col));
                        double curValue = getValue(row, col);
                        if (curValue < value) {
                            value = curValue;
                            index[0] = row;
                            index[1] = col;
                        }
                    }
                    else {
//                        double curValue = Math.max(value, getValue(row, col));
                        double curValue = getValue(row, col);
                        if (curValue > value) {
                            value = curValue;
                            index[0] = row;
                            index[1] = col;
                        }
                    }
                }
            }
        }
        else {
            for (int row = 0; row < getRows(); row++) {
                if (getRowMask(this, row)) {
                    for (int col = 0; col < getCols(); col++) {
                        if (!getMask(this, row, col) && getColMask(this, col)) {
                            if (min) {
                                double curValue = Math.min(value, getValue(row, col));
                                if (curValue < value) {
                                    value = curValue;
                                    index[0] = row;
                                    index[1] = col;
                                }
                            }
                            else {
                                double curValue = Math.max(value, getValue(row, col));
                                if (curValue > value) {
                                    value = curValue;
                                    index[0] = row;
                                    index[1] = col;
                                }
                            }
                        }
                    }
                }
            }
        }
        return value;
    }

    /**
     * Returns minimum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return minimum value of matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public double min() throws MatrixException {
        return argMinMax(true, new int[2]);
    }

    /**
     * Returns argmin meaning row and column of matrix containing minimum value.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return array containing row and column in this order that points to minimum value of matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public int[] argmin() throws MatrixException {
        int[] index = new int[2];
        argMinMax(true, index);
        return index;
    }

    /**
     * Returns maximum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return maximum value of matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public double max() throws MatrixException {
        return argMinMax(false, new int[2]);
    }

    /**
     * Returns argmax meaning row and column of matrix containing maximum value.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return array containing row and column in this order that points to maximum value of matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public int[] argmax() throws MatrixException {
        int[] index = new int[2];
        argMinMax(false, index);
        return index;
    }

    /**
     * Sets stride size for convolve and crosscorrelate operation.
     *
     * @param stride stride size.
     */
    public void setStride(int stride) {
        this.stride = stride;
    }

    /**
     * Returns stride size for convolve and crosscorrelate operation.
     *
     * @return stride size.
     */
    public int getStride() {
        return stride;
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated value of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix convolve(Matrix filter) throws MatrixException {
        Matrix result = new DMatrix(getRows() - filter.getRows() + 1, getCols() - filter.getCols() + 1);
        convolve(filter, result);
        return result;
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated value of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void convolve(Matrix filter, Matrix result) throws MatrixException {
        ProcedureFactory currentProcedureFactory = getProcedureFactory(filter, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createConvolveExpression(this, filter, result, stride, normalizers);
        int sliceRows = filter.getRows();
        int sliceCols = filter.getCols();
        for (int row = 0; row < result.getRows(); row += stride) {
            for (int col = 0; col < result.getCols(); col += stride) {
                convolve(filter, result, row, col, sliceRows, sliceCols);
            }
        }
    }

    /**
     * Calculates convolution between this matrix and filter matrix for a specific slice row and column.
     *
     * @param filter filter matrix.
     * @param result calculated value of convolution.
     */
    private void convolve(Matrix filter, Matrix result, int rowAt, int colAt, int sliceRows, int sliceCols) {
        double resultValue = 0;
        if (getMask() == null && filter.getMask() == null) {
            for (int row = 0; row < sliceRows; row++) {
                for (int col = 0; col < sliceCols; col++) {
                    resultValue += getValue(rowAt + row, colAt + col) * filter.getValue(sliceRows - 1 - row, sliceCols - 1 - col);
                }
            }
            result.setValue(rowAt, colAt, resultValue);
        }
        else {
            for (int row = 0; row < sliceRows; row++) {
                if (getRowMask(this, row) && getRowMask(filter, row)) {
                    for (int col = 0; col < sliceCols; col++) {
                        if (!getMask(this, row, col) && getColMask(this, col) && !getMask(filter, row, col) && getColMask(filter, col)) {
                            resultValue += getValue(rowAt + row, colAt + col) * filter.getValue(sliceRows - 1 - row, sliceCols - 1 - col);
                        }
                    }
                }
            }
            result.setValue(rowAt, colAt, resultValue);
        }
    }

    /**
     * Calculates gradient of convolution for output.
     *
     * @param filter filter for convolutional operator.
     * @return output gradient.
     */
    public Matrix convolveOutGrad(Matrix filter) {
        Matrix result = new DMatrix(getRows() + filter.getRows() - 1, getCols() + filter.getCols() - 1);
        convolveOutGrad(filter, result);
        return result;
    }

    /**
     * Calculates gradient of convolution for output.
     *
     * @param filter filter for convolutional operator.
     * @param result output gradient.
     */
    public void convolveOutGrad(Matrix filter, Matrix result) {
        int rows = getRows();
        int cols = getCols();
        for (int row = 0; row < rows; row += stride) {
            for (int col = 0; col < cols; col += stride) {
                convolveGrad(filter, result, row, col, true, filter.getRows(), filter.getCols());
            }
        }
    }

    /**
     * Calculates gradient of convolution for filter.
     *
     * @param input input for convolutional operator.
     * @return filter gradient.
     */
    public Matrix convolveFilterGrad(Matrix input) {
        Matrix result = new DMatrix(input.getRows() - getRows() + 1, input.getCols() - getCols() + 1);
        convolveFilterGrad(input, result);
        return result;
    }

    /**
     * Calculates gradient of convolution for filter.
     *
     * @param input input for convolutional operator.
     * @param resultGrad filter gradient.
     */
    public void convolveFilterGrad(Matrix input, Matrix resultGrad) {
        int rows = getRows();
        int cols = getCols();
        for (int row = 0; row < rows; row += stride) {
            for (int col = 0; col < cols; col += stride) {
                convolveGrad(input, resultGrad, row, col, false, resultGrad.getRows(), resultGrad.getCols());
            }
        }
    }

    /**
     * Calculates gradient slice for convolution operation. This matrix is output gradient for calculation.
     *
     * @param input input matrix for calculation.
     * @param result resulting gradient matrix.
     * @param gradRowAt gradient row.
     * @param gradColAt gradient column.
     * @param slideResult if true slides over result otherwise slides over input.
     * @param sliceRows size of slice (filter) in rows.
     * @param sliceCols size of slice (filter) in columns.
     */
    private void convolveGrad(Matrix input, Matrix result, int gradRowAt, int gradColAt, boolean slideResult, int sliceRows, int sliceCols) {
        int inputRowAt = slideResult ? 0 : gradRowAt;
        int inputColAt = slideResult ? 0 : gradColAt;
        int resultGradRowAt = slideResult ? gradRowAt : 0;
        int resultGradColAt = slideResult ? gradColAt : 0;
        double gradValue = getValue(gradRowAt, gradColAt);
        if (getMask() == null && input.getMask() == null) {
            for (int row = 0; row < sliceRows; row++) {
                for (int col = 0; col < sliceCols; col++) {
                    result.addValue(resultGradRowAt + row, resultGradColAt + col, input.getValue(inputRowAt + sliceRows - 1 - row, inputColAt + sliceCols - 1 - col) * gradValue);
                }
            }
        }
        else {
            for (int row = 0; row < sliceRows; row++) {
                if (getRowMask(this, row) && getRowMask(input, sliceRows - 1 - row)) {
                    for (int col = 0; col < sliceCols; col++) {
                        if (!getMask(this, row, col) && getColMask(this, col) && !getMask(input, sliceRows - 1 - row, col) && getColMask(input, sliceCols - 1 - col)) {
                            result.addValue(resultGradRowAt + row, resultGradColAt + col, input.getValue(inputRowAt + sliceRows - 1 - row, inputColAt + sliceCols - 1 - col) * gradValue);
                        }
                    }
                }
            }
        }
    }

    /**
     * Calculates cross-correlation between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated value of cross-correlation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix crosscorrelate(Matrix filter) throws MatrixException {
        Matrix result = new DMatrix(getRows() - filter.getRows() + 1, getCols() - filter.getCols() + 1);
        convolve(filter, result);
        return result;
    }

    /**
     * Calculates cross-correlation between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated value of cross-correlation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void crosscorrelate(Matrix filter, Matrix result) throws MatrixException {
        ProcedureFactory currentProcedureFactory = getProcedureFactory(filter, result);
        if (currentProcedureFactory != null) currentProcedureFactory.createCrosscorrelateExpression(this, filter, result, stride, normalizers);
        int sliceRows = filter.getRows();
        int sliceCols = filter.getCols();
        for (int row = 0; row < result.getRows(); row += stride) {
            for (int col = 0; col < result.getCols(); col += stride) {
                crosscorrelate(filter, result, row, col, sliceRows, sliceCols);
            }
        }
    }

    /**
     * Calculates cross-correlation between this matrix and filter matrix for a specific slice row and column.
     *
     * @param filter filter matrix.
     * @param result calculated value of cross-correlation.
     */
    private void crosscorrelate(Matrix filter, Matrix result, int rowAt, int colAt, int sliceRows, int sliceCols) {
        double resultValue = 0;
        if (getMask() == null && filter.getMask() == null) {
            for (int row = 0; row < sliceRows; row++) {
                for (int col = 0; col < sliceCols; col++) {
                    resultValue += getValue(rowAt + row, colAt + col) * filter.getValue(row,  col);
                }
            }
            result.setValue(rowAt, colAt, resultValue);
        }
        else {
            for (int row = 0; row < sliceRows; row++) {
                if (getRowMask(this, row) && getRowMask(filter, row)) {
                    for (int col = 0; col < sliceCols; col++) {
                        if (!getMask(this, row, col) && getColMask(this, col) && !getMask(filter, row, col) && getColMask(filter, col)) {
                            resultValue += getValue(rowAt + row, colAt + col) * filter.getValue(row, col);
                        }
                    }
                }
            }
            result.setValue(rowAt, colAt, resultValue);
        }

    }

    /**
     * Calculates gradient of cross-correlation for output.
     *
     * @param filter filter for cross-correlation operator.
     * @return output gradient.
     */
    public Matrix crosscorrelateOutGrad(Matrix filter) {
        Matrix result = new DMatrix(getRows() + filter.getRows() - 1, getCols() + filter.getCols() - 1);
        crosscorrelateOutGrad(filter, result);
        return result;
    }

    /**
     * Calculates gradient of cross-correlation for output.
     *
     * @param filter filter for cross-correlation operation.
     * @param result resulting output gradient.
     */
    public void crosscorrelateOutGrad(Matrix filter, Matrix result) {
        int rows = getRows();
        int cols = getCols();
        for (int row = 0; row < rows; row += stride) {
            for (int col = 0; col < cols; col += stride) {
                crosscorrelateGrad(filter, result, row, col, true, filter.getRows(), filter.getCols());
            }
        }
    }

    /**
     * Calculates gradient of cross-correlation for filter.
     *
     * @param input input for cross-correlation operator.
     * @return filter gradient.
     */
    public Matrix crosscorrelateFilterGrad(Matrix input) {
        Matrix result = new DMatrix(input.getRows() - getRows() + 1, input.getCols() - getCols() + 1);
        crosscorrelateFilterGrad(input, result);
        return result;
    }

    /**
     * Calculates gradient of cross-correlation for filter.
     *
     * @param input input for cross-correlation operation.
     * @param resultGrad resulting filter gradient.
     */
    public void crosscorrelateFilterGrad(Matrix input, Matrix resultGrad) {
        int rows = getRows();
        int cols = getCols();
//        setStride(input.getStride());
        for (int row = 0; row < rows; row += stride) {
            for (int col = 0; col < cols; col += stride) {
                crosscorrelateGrad(input, resultGrad, row, col, false, resultGrad.getRows(), resultGrad.getCols());
            }
        }
    }

    /**
     * Calculates gradient slice for cross-correlation operation. This matrix is output gradient for calculation.
     *
     * @param input input matrix for calculation.
     * @param result resulting gradient matrix.
     * @param gradRowAt gradient row.
     * @param gradColAt gradient column.
     * @param slideResult if true slides over result otherwise slides over input.
     * @param sliceRows size of slice (filter) in rows.
     * @param sliceCols size of slice (filter) in columns.
     */
    private void crosscorrelateGrad(Matrix input, Matrix result, int gradRowAt, int gradColAt, boolean slideResult, int sliceRows, int sliceCols) {
        int inputRowAt = slideResult ? 0 : gradRowAt;
        int inputColAt = slideResult ? 0 : gradColAt;
        int resultGradRowAt = slideResult ? gradRowAt : 0;
        int resultGradColAt = slideResult ? gradColAt : 0;
        double gradValue = getValue(gradRowAt, gradColAt);
        if (getMask() == null && input.getMask() == null) {
            for (int row = 0; row < sliceRows; row++) {
                for (int col = 0; col < sliceCols; col++) {
                    result.addValue(resultGradRowAt + row, resultGradColAt + col, input.getValue(inputRowAt + row, inputColAt + col) * gradValue);
                }
            }
        }
        else {
            for (int row = 0; row < sliceRows; row++) {
                if (getRowMask(this, row) && getRowMask(input, row)) {
                    for (int col = 0; col < sliceCols; col++) {
                        if (!getMask(this, row, col) && getColMask(this, col) && !getMask(input, row, col) && getColMask(input, col)) {
                            result.addValue(resultGradRowAt + row, resultGradColAt + col, input.getValue(inputRowAt + row, inputColAt + col) * gradValue);
                        }
                    }
                }
            }
        }
    }

    /**
     * Sets size of pool for pooling operation.
     *
     * @param poolSize pool size.
     */
    public void setPoolSize(int poolSize) {
        this.poolSize = poolSize;
    }

    /**
     * Returns pool size.
     *
     * @return pool size.
     */
    public int getPoolSize() {
        return poolSize;
    }

    /**
     * Calculates max pooling operation for this matrix and returns max arguments.
     *
     * @param maxArgsAt arguments on maximum row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix maxPool(int [][][] maxArgsAt) throws MatrixException {
        Matrix result = new DMatrix(getRows() - poolSize + 1, getCols() - poolSize + 1);
        maxPool(result, maxArgsAt);
        return result;
    }

    /**
     * Calculates max pooling operation for this matrix and returns max arguments.
     *
     * @param result result matrix.
     * @param maxArgsAt arguments on maximum row and col value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void maxPool(Matrix result, int [][][] maxArgsAt) throws MatrixException {
        ProcedureFactory currentProcedureFactory = getProcedureFactory(result);
        if (currentProcedureFactory != null) currentProcedureFactory.createMaxPoolExpression(this, result, stride, poolSize, normalizers);
        for (int row = 0; row < result.getRows(); row += stride) {
            for (int col = 0; col < result.getCols(); col += stride) {
                maxPool(result, row, col, maxArgsAt);
            }
        }
    }

    /**
     * Calculates max pooling operation for this matrix and returns max arguments.
     *
     * @param result result matrix.
     * @param maxArgsAt arguments on maximum row and col value.
     */
    private void maxPool(Matrix result, int rowAt, int colAt, int [][][] maxArgsAt) {
        double maxValue = Double.NEGATIVE_INFINITY;
        if (getMask() == null) {
            for (int row = 0; row < poolSize; row++) {
                for (int col = 0; col < poolSize; col++) {
                    int currentRow = rowAt + row;
                    int currentCol = colAt + col;
                    double curValue = getValue(currentRow, currentCol);
                    if (maxValue < curValue) {
                        maxValue = curValue;
                        maxArgsAt[rowAt][colAt][0] = currentRow;
                        maxArgsAt[rowAt][colAt][1] = currentCol;
                    }
                }
            }
            result.setValue(rowAt, colAt, maxValue);
        }
        else {
            for (int row = 0; row < poolSize; row++) {
                if (getRowMask(this, row)) {
                    for (int col = 0; col < poolSize; col++) {
                        if (!getMask(this, row, col) && getColMask(this, col)) {
                            int currentRow = rowAt + row;
                            int currentCol = colAt + col;
                            double curValue = getValue(currentRow, currentCol);
                            if (maxValue < curValue) {
                                maxValue = curValue;
                                maxArgsAt[rowAt][colAt][0] = currentRow;
                                maxArgsAt[rowAt][colAt][1] = currentCol;
                            }
                        }
                    }
                }
            }
            result.setValue(rowAt, colAt, maxValue);
        }
    }

    /**
     * Calculates gradient of max pooling operation for this matrix.
     *
     * @param maxArgsAt arguments on maximum row and col value.
     * @return result matrix.
     */
    public Matrix maxPoolGrad(int [][][] maxArgsAt) {
        Matrix result = new DMatrix(getRows() + poolSize - 1, getCols() + poolSize - 1);
        maxPoolGrad(result, maxArgsAt);
        return result;
    }

    /**
     * Calculates gradient for max pool operation.
     *
     * @param result result matrix.
     * @param maxArgsAt arguments on maximum row and col value.
     */
    public void maxPoolGrad(Matrix result, int[][][] maxArgsAt) {
        for (int row = 0; row < getRows(); row += stride) {
            for (int col = 0; col < getCols(); col += stride) {
                maxPoolGrad(result, row, col, maxArgsAt[row][col]);
            }
        }
    }

    /**
     * Calculates gradient for max pool operation at certain row and column position.
     *
     * @param result result matrix.
     * @param rowAt row position.
     * @param colAt column position.
     * @param maxArgsAt arguments on maximum row and col value.
     */
    private void maxPoolGrad(Matrix result, int rowAt, int colAt, int[] maxArgsAt) {
        if (getMask() == null) result.setValue(maxArgsAt[0], maxArgsAt[1], getValue(rowAt, colAt));
        else {
            if (getRowMask(this, rowAt) && !getMask(this, rowAt, colAt) && getColMask(this, colAt)) {
                result.setValue(maxArgsAt[0], maxArgsAt[1],  getValue(rowAt, colAt));
            }
        }
    }

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix avgPool() throws MatrixException {
        Matrix result = new DMatrix(getRows() - poolSize + 1, getCols() - poolSize + 1);
        avgPool(result);
        return result;
    }

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void avgPool(Matrix result) throws MatrixException {
        ProcedureFactory currentProcedureFactory = getProcedureFactory(result);
        if (currentProcedureFactory != null) currentProcedureFactory.createAveragePoolExpression(this, result, stride, poolSize, normalizers);
        for (int row = 0; row < result.getRows(); row += stride) {
            for (int col = 0; col < result.getCols(); col += stride) {
                avgPool(result, row, col);
            }
        }
    }

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @param result result matrix.
     */
    private void avgPool(Matrix result, int rowAt, int colAt) {
        double sumValue = 0;
        if (getMask() == null) {
            for (int row = 0; row < poolSize; row++) {
                for (int col = 0; col < poolSize; col++) {
                    sumValue += getValue(rowAt + row, colAt + col);
                }
            }
            result.setValue(rowAt, colAt, sumValue / (poolSize * poolSize));
        }
        else {
            for (int row = 0; row < poolSize; row++) {
                if (getRowMask(this, row)) {
                    for (int col = 0; col < poolSize; col++) {
                        if (!getMask(this, row, col) && getColMask(this, col)) {
                            sumValue += getValue(rowAt + row, colAt + col);
                        }
                    }
                }
            }
            result.setValue(rowAt, colAt, sumValue / (poolSize * poolSize));
        }
    }

    /**
     * Calculates gradient of average pooling operation for this matrix.
     *
     * @return result matrix.
     */
    public Matrix avgPoolGrad() {
        Matrix result = new DMatrix(getRows() + poolSize - 1, getCols() + poolSize - 1);
        avgPoolGrad(result);
        return result;
    }

    /**
     * Calculates gradient of average pooling operation for this matrix.
     *
     * @param result result matrix.
     */
    public void avgPoolGrad(Matrix result) {
        for (int row = 0; row < result.getRows(); row += stride) {
            for (int col = 0; col < result.getCols(); col += stride) {
                avgPoolGrad(result, row, col);
            }
        }
    }

    /**
     * Calculates gradient of average pooling operation for this matrix.
     *
     * @param result result matrix.
     */
    private void avgPoolGrad(Matrix result, int rowAt, int colAt) {
        double value = 1 / (double)(poolSize * poolSize);
        if (getMask() == null) {
            for (int row = 0; row < poolSize; row++) {
                for (int col = 0; col < poolSize; col++) {
                    result.setValue(rowAt + row, colAt + col, value);
                }
            }
        }
        else {
            for (int row = 0; row < poolSize; row++) {
                if (getRowMask(this, row)) {
                    for (int col = 0; col < poolSize; col++) {
                        if (!getMask(this, row, col) && getColMask(this, col)) {
                            result.setValue(rowAt + row, colAt + col, value);
                        }
                    }
                }
            }
        }
    }

    /**
     * Transposes matrix.
     *
     * @return reference to this matrix but with transposed that is flipped rows and columns.
     * @throws MatrixException throws exception if mask operation fails.
     */
    public Matrix T() throws MatrixException {
        try {
            // Make shallow copy of matrix leaving references internal objects which are shared.
            Matrix clone = (Matrix)clone();
            clone.t = !clone.t; // transpose
            if (getMask() != null) clone.setMask(getMask().T());
            return clone;
        } catch (CloneNotSupportedException exception) {
            System.out.println("Matrix cloning failed");
        }
        return null;
    }

    /**
     * Checks if matrix is transposed.
     *
     * @return true is matrix is transposed otherwise false.
     */
    public boolean isT() {
        return t;
    }

    /**
     * Concatenates this and other matrix vertically.
     *
     * @param other matrix to be concatenated to the end of this matrix vertically.
     * @throws MatrixException throws exception if column dimensions of this and other matrix are not matching.
     */
    public void concatenateVertical(Matrix other) throws MatrixException {
        if (getCols() != other.getCols()) {
            throw new MatrixException("Merge Vertical: Incompatible matrix sizes: " + getRows() + "x" + getCols() + " by " + other.getRows() + "x" + other.getCols());
        }
        Matrix newMatrix = getNewMatrix(getRows() + other.getRows(), getCols());
        for (int row = 0; row < getRows(); row++) {
            for (int col = 0; col < getCols(); col++) {
                newMatrix.setValue(row, col, getValue(row, col));
            }
        }
        for (int row = 0; row < other.getRows(); row++) {
            for (int col = 0; col < other.getCols(); col++) {
                newMatrix.setValue(getRows() + row, col, other.getValue(row, col));
            }
        }
        setAsMatrix(newMatrix);
    }

    /**
     * Concatenates this and other matrix horizontally.
     *
     * @param other matrix to be concatenated to the end of this matrix horizontally.
     * @throws MatrixException throws exception if row dimensions of this and other matrix are not matching.
     */
    public void concatenateHorizontal(Matrix other) throws MatrixException {
        if (getRows() != other.getRows()) {
            throw new MatrixException("Merge Horizontal: Incompatible matrix sizes: " + getRows() + "x" + getCols() + " by " + other.getRows() + "x" + other.getCols());
        }
        Matrix newMatrix = getNewMatrix(getRows(), getCols() + other.getCols());
        for (int row = 0; row < getRows(); row++) {
            for (int col = 0; col < getCols(); col++) {
                newMatrix.setValue(row, col, getValue(row, col));
            }
        }
        for (int row = 0; row < other.getRows(); row++) {
            for (int col = 0; col < other.getCols(); col++) {
                newMatrix.setValue(row, getCols() + col, other.getValue(row, col));
            }
        }
        setAsMatrix(newMatrix);
    }

    /**
     * Prints matrix in row and column format.
     *
     */
    public void print() {
        for (int row = 0; row < getRows(); row++) {
            System.out.print("[");
            for (int col = 0; col < getCols(); col++) {
                System.out.print(getValue(row, col));
                if (col < getCols() - 1) System.out.print(" ");
            }
            System.out.println("]");
        }
    }

    /**
     * Prints size (rows x columns) of matrix.
     *
     */
    public void printSize() {
        System.out.println("Matrix size: " + getRows() + "x" + getCols());
    }

    /**
     * Sets mask to this matrix.
     *
     * @param newMask new mask as input.
     * @throws MatrixException throws exception if new mask dimensions or mask type are not matching with this mask.
     */
    public void setMask(Mask newMask) throws MatrixException {
        if (getRows() != newMask.getRows() || getCols() != newMask.getCols()) throw new MatrixException("Dimensions of new mask are not matching with matrix dimensions.");
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
     * Returns mask at specific position.
     *
     * @param matrix matrix as input.
     * @param row specific row.
     * @param col specific column.
     * @return if true mask exists and is masked at specific position (row + column).
     */
    public static boolean getMask(Matrix matrix, int row, int col) {
        return matrix.getMask() != null && matrix.getMask().getMask(row, col);
    }

    /**
     * Returns mask at specific row.
     *
     * @param matrix matrix as input.
     * @param row specific row.
     * @return if true mask exists and is masked at specific row.
     */
    public static boolean getRowMask(Matrix matrix, int row) {
        return matrix.getMask() == null || !matrix.getMask().getRowMask(row);
    }

    /**
     * Returns mask at specific column.
     *
     * @param matrix matrix as input.
     * @param col specific column.
     * @return if true mask exists and is masked at specific column.
     */
    public static boolean getColMask(Matrix matrix, int col) {
        return matrix.getMask() == null || !matrix.getMask().getColMask(col);
    }

    /**
     * Returns new mask for this matrix.<br>
     * Implemented by underlying matrix class.<br>
     *
     * @return mask of this matrix.
     */
    protected abstract Mask getNewMask();

    /**
     * Sets scaling constant that scales (multiplies) each matrix element by this constant.<br>
     * Default value is 1.<br>
     *
     * @param scalingConstant scaling constant to be set.
     */
    public void setScalingConstant(double scalingConstant) {
        this.scalingConstant = scalingConstant;
    }

    /**
     * Unsets scaling constant for matrix (resets it to default value 1).
     *
     */
    public void unsetScalingConstant() {
        this.scalingConstant = 1;
    }

}
