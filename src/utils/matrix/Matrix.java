/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.matrix;

import utils.DynamicParamException;
import utils.procedure.ProcedureFactory;

import java.io.Serializable;
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
         * @param column column of matrix if relevant for initialization.
         * @return value to be used for initialization.
         */
        double value(int row, int column);
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
     *
     */
    protected boolean isTransposed;

    /**
     * Initializer variable.
     *
     */
    private Initializer initializer;

    /**
     * Initialization type
     *
     */
    private Initialization initialization = Initialization.ZERO;

    /**
     * Reference to mask of matrix. If null mask is not used.
     *
     */
    private Mask mask;

    /**
     * Scaling constant applied in all matrix operations meaning scalingConstant * operation.<br>
     * If constant is 1 effectively no scaling is done.<br>
     *
     */
    private double scalingConstant = 1;

    /**
     * If true matrix is treated as scalar (1x1) matrix otherwise as normal matrix.
     *
     */
    private final boolean isScalar;

    /**
     * Stride size for convolutional and pooling operations.
     *
     */
    private int stride;

    /**
     * Dilation step size for convolutional operations.
     *
     */
    private int dilation;

    /**
     * Filter size for convolutional operations.
     *
     */
    private int filterSize;

    /**
     * Pool size for pooling operations.
     *
     */
    private int poolSize;

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
     * Random function for matrix class.
     *
     */
    private final Random random = new Random();

    /**
     * Name of matrix.
     *
     */
    protected String name;

    /**
     * Constructor for matrix.
     *
     * @param isScalar true if matrix is scalar (size 1x1).
     */
    protected Matrix(boolean isScalar) {
        this.isScalar = isScalar;
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
     * Abstract matrix reset function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
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
     * @param standardDeviation standard deviation of the distribution.
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
        this.initialization = initialization;
        switch (initialization) {
            case ZERO:
                initializer = (Initializer & Serializable) (row, col) -> 0;
                break;
            case ONE:
                initializer = (Initializer & Serializable) (row, col) -> 1;
                initialize(initializer);
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
                initializer = (Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / (double)(getRows() + getColumns())));
                initialize(initializer);
                break;
            case UNIFORM_XAVIER:
                initializer = (Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(getRows() + getColumns())));
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
     * Returns true if matrix is scalar otherwise false.
     *
     * @return true if matrix is scalar otherwise false.
     */
    public boolean isScalar() {
        return isScalar;
    }

    /**
     * Returns matrix initialization type.
     *
     * @return current matrix initialization type.
     */
    private Initialization getInitialization() {
        return initialization;
    }

    /**
     * Initializes matrix with given initializer operation.
     *
     * @param initializer initializer operation.
     */
    public void initialize(Initializer initializer) {
        for (int row = 0; row < getRows(); row++) {
            for (int col = 0; col < getColumns(); col++) {
                setValue(row, col, initializer.value(row, col));
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
            for (int col = 0; col < getColumns(); col++) {
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
     * @param column column of value to be set.
     * @param value new value to be set.
     */
    public abstract void setValue(int row, int column, double value);

    /**
     * Matrix internal function used to get value of specific row and column.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @return value of row and column.
     */
    public abstract double getValue(int row, int column);

    /**
     * Returns size (rows * columns) of matrix.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @return size of matrix.
     */
    public abstract int size();

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
    public abstract int getColumns();

    /**
     * Matrix function used to add value of specific row and column.
     *
     * @param row row of value to be added.
     * @param column column of value to be added.
     * @param value to be added.
     */
    public void incrementByValue(int row, int column, double value) {
        setValue(row, column, getValue(row, column) + value);
    }

    /**
     * Matrix function used to decrease value of specific row and column.
     *
     * @param row row of value to be decreased.
     * @param column column of value to be decreased.
     * @param value to be decreased.
     */
    public void decrementByValue(int row, int column, double value) {
        setValue(row, column, getValue(row, column) - value);
    }

    /**
     * Matrix function used to multiply value of specific row and column.
     *
     * @param row row of value to be multiplied.
     * @param column column of value to be multiplied.
     * @param value to be multiplied.
     */
    public void multiplyByValue(int row, int column, double value) {
        setValue(row, column, getValue(row, column) * value);
    }

    /**
     * Matrix function used to divide value of specific row and column.
     *
     * @param row row of value to be divided.
     * @param column column of value to be divided.
     * @param value to be divided.
     */
    public void divideByValue(int row, int column, double value) {
        setValue(row, column, getValue(row, column) / value);
    }

    /**
     * Returns new matrix of dimensions rows x columns.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @return new matrix of dimensions rows x columns.
     */
    public abstract Matrix getNewMatrix();

    /**
     * Copies new matrix inside this matrix with dimensions rows x columns.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @param newMatrix new matrix to be copied inside this matrix.
     */
    protected abstract void copyMatrixData(Matrix newMatrix);

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
            newMatrix.copyMatrixData(this);
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
     * Makes current matrix data equal to other matrix data.
     *
     * @param other other matrix to be copied as data of this matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void setEqualTo(Matrix other) throws MatrixException {
        if (other.getRows() != getRows() || other.getColumns() != getColumns()) {
            throw new MatrixException("Incompatible target matrix size: " + other.getRows() + "x" + other.getColumns());
        }
        for (int row = 0; row < other.getRows(); row++) {
            for (int column = 0; column < other.getColumns(); column++) {
                setValue(row, column, other.getValue(row, column));
                if (getMask() !=null) getMask().setMask(row, column, hasMaskAt(other, row, column));
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
        if (other.getRows() != getRows() || other.getColumns() != getColumns()) {
            throw new MatrixException("Incompatible target matrix size: " + other.getRows() + "x" + other.getColumns());
        }
        for (int row = 0; row < other.getRows(); row++) {
            for (int column = 0; column < other.getColumns(); column++) {
                if (getValue(row, column) != other.getValue(row, column)) return false;
            }
        }
        return true;
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
     * Applies single variable operation to this matrix and stores operation result into result matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param result matrix which stores operation result.
     * @param matrixUnaryOperation single variable operation defined as lambda operator.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and result matrix are not of equal dimensions.
     */
    public Matrix apply(Matrix result, MatrixUnaryOperation matrixUnaryOperation) throws MatrixException {
        if (result.getRows() != getRows() || result.getColumns() != getColumns()) {
            throw new MatrixException("Incompatible result matrix sizes: " + result.getRows() + "x" + result.getColumns());
        }
        if (getMask() == null) {
            for (int row = 0; row < getRows(); row++) {
                for (int column = 0; column < getColumns(); column++) {
                    result.setValue(row, column, scalingConstant * matrixUnaryOperation.execute(getValue(row, column)));
                }
            }
        }
        else {
            for (int row = 0; row < getRows(); row++) {
                if (!hasRowMaskAt(this, row)) {
                    for (int column = 0; column < getColumns(); column++) {
                        if (!hasMaskAt(this, row, column) && !hasColumnMaskAt(this, column)) {
                            result.setValue(row, column, scalingConstant * matrixUnaryOperation.execute(getValue(row, column)));
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
     * @param matrixUnaryOperation single variable operation defined as lambda operator.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix apply(MatrixUnaryOperation matrixUnaryOperation) throws MatrixException {
        return apply(getNewMatrix(), matrixUnaryOperation);
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
        double expressionLock = 0;
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        apply(result, unaryFunction.getFunction());
        if (procedureFactory != null) procedureFactory.createUnaryFunctionExpression(expressionLock, this, result, unaryFunction);
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
        Matrix result = apply(getNewMatrix(), unaryFunction.getFunction());
        apply(result, unaryFunction);
        return result;
    }

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param unaryFunctionType unaryFunction type to be applied.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void apply(Matrix result, UnaryFunctionType unaryFunctionType) throws MatrixException, DynamicParamException {
        apply(result, new UnaryFunction(unaryFunctionType));
    }

    /**
     * Applies unaryFunction to this matrix and return operation result.<br>
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
     * Applies two variable operation to this matrix and other matrix and stores operation result into result matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix or
     * multiplying current matrix with other matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @param matrixBinaryOperation two variable operation defined as lambda operator.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     */
    public Matrix applyBi(Matrix other, Matrix result, Matrix.MatrixBinaryOperation matrixBinaryOperation) throws MatrixException {
        if (!isScalar() && !other.isScalar() && (getRows() != other.getRows() || getColumns() != other.getColumns())) {
            throw new MatrixException("Incompatible matrix sizes: " + getRows() + "x" + getColumns() + " by " + other.getRows() + "x" + other.getColumns());
        }
        if (!isScalar() && !result.isScalar() && (getRows() != result.getRows() || getColumns() != result.getColumns())) {
            throw new MatrixException("Incompatible result matrix sizes: " + result.getRows() + "x" + result.getColumns());
        }
        int rows = !isScalar() ? getRows() : other.getRows();
        int columns = !isScalar() ? getColumns() : other.getColumns();
        if (getMask() == null && other.getMask() == null) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    result.setValue(row, column, scalingConstant * matrixBinaryOperation.execute(getValue(row, column), other.getValue(row, column)));
                }
            }
        }
        else {
            for (int row = 0; row < rows; row++) {
                if (!hasRowMaskAt(this, row) && !hasRowMaskAt(other, row)) {
                    for (int column = 0; column < columns; column++) {
                        if (!hasMaskAt(this, row, column) && !hasColumnMaskAt(this, column) && !hasMaskAt(other, row, column) && !hasColumnMaskAt(other, column)) {
                            result.setValue(row, column, scalingConstant * matrixBinaryOperation.execute(getValue(row, column), other.getValue(row, column)));
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * Applies two variable operation to this matrix and other matrix and stores operation result into result matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix or
     * multiplying current matrix with other matrix.<br>
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
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        applyBi(other, result, binaryFunction.getFunction());
        if (procedureFactory != null) procedureFactory.createBinaryFunctionExpression(expressionLock, this, other, result, binaryFunction);
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
        Matrix result = getResultMatrix(other);
        applyBi(other, result, binaryFunction);
        return result;
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
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void applyBi(Matrix other, Matrix result, BinaryFunctionType binaryFunctionType) throws MatrixException, DynamicParamException {
        applyBi(other, result, new BinaryFunction(binaryFunctionType));
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
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) Double::sum);
        if (procedureFactory != null) procedureFactory.createAddExpression(expressionLock, this, other, result);
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
        double expressionLock = 0;
        Matrix other = new DMatrix(constant);
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        add(other, result);
        if (procedureFactory != null) procedureFactory.createAddExpression(expressionLock, this, other, result);
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
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 - value2);
        if (procedureFactory != null) procedureFactory.createSubtractExpression(expressionLock, this, other, result);
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
        double expressionLock = 0;
        Matrix other = new DMatrix(constant);
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        subtract(other, result);
        if (procedureFactory != null) procedureFactory.createSubtractExpression(expressionLock, this, other, result);
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
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 * value2);
        if (procedureFactory != null) procedureFactory.createMultiplyExpression(expressionLock, this, other, result);
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
        double expressionLock = 0;
        Matrix other = new DMatrix(constant);
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        multiply(other, result);
        if (procedureFactory != null) procedureFactory.createMultiplyExpression(expressionLock, this, other, result);
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
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        applyBi (other, result, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value2 != 0 ? value1 / value2 : Double.POSITIVE_INFINITY);
        if (procedureFactory != null) procedureFactory.createDivideExpression(expressionLock, this, other, result);
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
        double expressionLock = 0;
        Matrix other = new DMatrix(constant);
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        divide(other, result);
        if (procedureFactory != null) procedureFactory.createDivideExpression(expressionLock, this, other, result);
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
        return applyBi (asMatrix(power), BinaryFunctionType.POW);
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
        applyBi (asMatrix(power), result, BinaryFunctionType.POW);
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
        if (getColumns() != other.getRows()) {
            throw new MatrixException("Incompatible matrix sizes: " + getRows() + "x" + getColumns() + " by " + other.getRows() + "x" + other.getColumns());
        }
        if (getRows() != result.getRows() || other.getColumns() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        if (getMask() == null && other.getMask() == null) {
            for (int row = 0; row < getRows(); row++) {
                for (int column = 0; column < other.getColumns(); column++) {
                    result.setValue(row, column, 0);
                    for (int x = 0; x < getColumns(); x++) {
                        result.setValue(row, column, result.getValue(row, column) + scalingConstant * getValue(row, x) * other.getValue(x, column));
                    }
                }
            }
        }
        else {
            for (int row = 0; row < getRows(); row++) {
                if (!hasRowMaskAt(this, row)) {
                    for (int column = 0; column < other.getColumns(); column++) {
                        if (!hasColumnMaskAt(other, column)) {
                            for (int x = 0; x < getColumns(); x++) {
                                if (!hasMaskAt(this, row, x) && !hasMaskAt(other, x, column)) {
                                    result.setValue(row, column, result.getValue(row, column) + scalingConstant * getValue(row, x) * other.getValue(x, column));
                                }
                            }
                        }
                    }
                }
            }
        }
        if (procedureFactory != null) procedureFactory.createDotExpression(expressionLock, this, other, result);
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
        return dot(other, new DMatrix(getRows(), other.getColumns()));
    }

    /**
     * Returns constant as constant matrix.
     *
     * @param constant constant value.
     * @return constant matrix.
     */
    public Matrix asMatrix(double constant) {
        Matrix constantMatrix = new DMatrix(constant);
        constantMatrix.setValue(0, 0, constant);
        return constantMatrix;
    }

    /**
     * Takes cumulative sum of single variable operation applied over each element of this matrix.<br>
     * Returns result array which has first element containing cumulative sum and second element number of elements.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param operation single variable operation defined as lambda operator.
     * @return array containing cumulative sum and element count as elements.
     */
    public double[] count(MatrixUnaryOperation operation) {
        double[] result = new double[2];
        result[0] = 0;
        result[1] = 0;
        if (getMask() == null) {
            for (int row = 0; row < getRows(); row++) {
                for (int column = 0; column < getColumns(); column++) {
                    result[0] += operation.execute(getValue(row, column));
                    result[1]++;
                }
            }
        }
        else {
            for (int row = 0; row < getRows(); row++) {
                if (!hasRowMaskAt(this, row)) {
                    for (int column = 0; column < getColumns(); column++) {
                        if (!hasMaskAt(this, row, column) && !hasColumnMaskAt(this, column)) {
                            result[0] += operation.execute(getValue(row, column));
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
     * Takes element wise cumulative sum of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return cumulative sum of this matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix sumAsMatrix() throws MatrixException {
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        Matrix result = asMatrix(sum());
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) procedureFactory.createSumExpression(expressionLock, this, result);
        return result;
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
     * Takes mean of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @throws MatrixException not thrown in any situation.
     * @return mean of elements of this matrix.
     */
    public Matrix meanAsMatrix() throws MatrixException {
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        Matrix result = asMatrix(mean());
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) procedureFactory.createMeanExpression(expressionLock, this, result);
        return result;
    }

    /**
     * Takes variance of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return variance of elements of this matrix.
     */
    public double variance() {
        return variance(mean());
    }

    /**
     * Takes variance of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @throws MatrixException not thrown in any situation.
     * @return variance of elements of this matrix.
     */
    public Matrix varianceAsMatrix() throws MatrixException {
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        Matrix result = asMatrix(variance());
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) procedureFactory.createVarianceExpression(expressionLock, this, result);
        return result;
    }

    /**
     * Takes variance of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return variance of elements of this matrix.
     */
    public double variance(double mean) {
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
    public Matrix varianceAsMatrix(Matrix mean) {
        return asMatrix(variance(mean.getValue(0, 0)));
    }

    /**
     * Takes standard deviation of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return standard deviation of elements of this matrix.
     */
    public double standardDeviation() {
        return standardDeviation(mean());
    }

    /**
     * Takes standard deviation of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @throws MatrixException not thrown in any situation.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @return standard deviation of elements of this matrix.
     */
    public Matrix standardDeviationAsMatrix() throws MatrixException, DynamicParamException {
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        Matrix result = new DMatrix(standardDeviation());
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) procedureFactory.createStandardDeviationExpression(expressionLock, this, result);
        return result;
    }

    /**
     * Takes standard deviation of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return standard deviation of elements of this matrix.
     */
    public double standardDeviation(double mean) {
        MatrixUnaryOperation operation = (Matrix.MatrixUnaryOperation & Serializable) value -> Math.pow(value - mean, 2);
        double[] result = count(operation);
        return result[1] > 1 ? Math.sqrt(result[0] / (result[1] - 1)) : 0;
    }

    /**
     * Takes standard deviation of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return standard deviation of elements of this matrix.
     */
    public Matrix standardDeviationAsMatrix(Matrix mean) {
        return asMatrix(standardDeviation(mean.getValue(0, 0)));
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
        return Math.pow(result[0], 1 / (double)p);
    }

    /**
     * Takes cumulative p- norm (p is number equal or bigger than 1) of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param p p value for norm.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return cumulative norm value of matrix.
     */
    public Matrix normAsMatrix(int p) throws MatrixException {
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        Matrix result = asMatrix(norm(p));
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) procedureFactory.createNormExpression(expressionLock, this, result, p);
        return result;
    }

    /**
     * Calculates exponential moving average.
     *
     * @param currentAverage current average value
     * @param newAverage new average value
     * @param beta degree of weighting decrease for exponential moving average.
     * @return updated average with new average value included.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public static Matrix exponentialMovingAverage(Matrix currentAverage, Matrix newAverage, double beta) throws MatrixException {
        return currentAverage == null ? newAverage : currentAverage.multiply(beta).add(newAverage.multiply(1 - beta));
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
        double standardDeviation = other.standardDeviation();
        return other.apply(other, (Matrix.MatrixUnaryOperation & Serializable) (value) -> (value - mean) / standardDeviation);
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
        Matrix.minMax(this, newMinimum, newMaximum);
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
    public static Matrix minMax(Matrix other, double newMinimum, double newMaximum) throws MatrixException {
        double minimum = other.min();
        double maximum = other.max();
        double delta = maximum - minimum != 0 ? maximum - minimum : 1;
        return other.apply(other, (Matrix.MatrixUnaryOperation & Serializable) (value) -> (value - minimum) / delta * (newMaximum - newMinimum) + newMinimum);
    }

    /**
     * Finds minimum or maximum element of matrix and return this value with row and column information.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param isMinimum If true finds minimum value with row and column information otherwise maximum value.
     * @param index two dimensional array used to return minimum or maximum value row and column in this order.
     * @return minimum or maximum value found.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public double argMinMax(boolean isMinimum, int[] index) throws MatrixException {
        if (index.length != 2) throw new MatrixException("Dimension of index must be 2.");
        double value = isMinimum ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        if (getMask() == null) {
            for (int row = 0; row < getRows(); row++) {
                for (int column = 0; column < getColumns(); column++) {
                    if (isMinimum) {
                        double currentValue = getValue(row, column);
                        if (currentValue < value) {
                            value = currentValue;
                            index[0] = row;
                            index[1] = column;
                        }
                    }
                    else {
                        double currentValue = getValue(row, column);
                        if (currentValue > value) {
                            value = currentValue;
                            index[0] = row;
                            index[1] = column;
                        }
                    }
                }
            }
        }
        else {
            for (int row = 0; row < getRows(); row++) {
                if (!hasRowMaskAt(this, row)) {
                    for (int column = 0; column < getColumns(); column++) {
                        if (!hasMaskAt(this, row, column) && !hasColumnMaskAt(this, column)) {
                            if (isMinimum) {
                                double currentValue = Math.min(value, getValue(row, column));
                                if (currentValue < value) {
                                    value = currentValue;
                                    index[0] = row;
                                    index[1] = column;
                                }
                            }
                            else {
                                double currentValue = Math.max(value, getValue(row, column));
                                if (currentValue > value) {
                                    value = currentValue;
                                    index[0] = row;
                                    index[1] = column;
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
     * Returns minimum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @throws MatrixException not thrown in any situation.
     * @return minimum value of matrix.
     */
    public Matrix minAsMatrix() throws MatrixException {
        return asMatrix(min());
    }

    /**
     * Returns argmin meaning row and column of matrix containing minimum value.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @throws MatrixException not thrown in any situation.
     * @return array containing row and column in this order that points to minimum value of matrix.
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
     * @throws MatrixException not thrown in any situation.
     * @return maximum value of matrix.
     */
    public double max() throws MatrixException {
        return argMinMax(false, new int[2]);
    }

    /**
     * Returns maximum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @throws MatrixException not thrown in any situation.
     * @return maximum value of matrix.
     */
    public Matrix maxAsMatrix() throws MatrixException {
        return asMatrix(max());
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
     * Returns softmax of this matrix.
     *
     * @param result result matrix.
     * @return softmax of this matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmax(Matrix result) throws MatrixException {
        if (getColumns() != 1) {
            throw new MatrixException("Matrix must be a column vector.");
        }
        if (getRows() != result.getRows() || getColumns() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }

        double maxValue = Double.NEGATIVE_INFINITY;
        double sumValue = 0;
        if (getMask() == null) {
            for (int row = 0; row < getRows(); row++) {
                maxValue = Math.max(maxValue, getValue(row, 0));
            }
            for (int row = 0; row < getRows(); row++) {
                double value = Math.exp(getValue(row, 0) - maxValue);
                sumValue += value;
                result.setValue(row, 0, value);
            }
            for (int row = 0; row < getRows(); row++) {
                result.setValue(row, 0, result.getValue(row, 0) / sumValue);
            }
        }
        else {
            for (int row = 0; row < getRows(); row++) {
                if (!hasRowMaskAt(this, row) && !hasMaskAt(this, row, 0) && !hasColumnMaskAt(this, 0)) {
                    maxValue = Math.max(maxValue, getValue(row, 0));
                }
            }
            for (int row = 0; row < getRows(); row++) {
                if (!hasRowMaskAt(this, row) && !hasMaskAt(this, row, 0) && !hasColumnMaskAt(this, 0)) {
                    double value = Math.exp(getValue(row, 0) - maxValue);
                    sumValue += value;
                    result.setValue(row, 0, value);
                }
            }
            for (int row = 0; row < getRows(); row++) {
                if (!hasRowMaskAt(this, row) && !hasMaskAt(this, row, 0) && !hasColumnMaskAt(this, 0)) {
                    result.setValue(row, 0, result.getValue(row, 0) / sumValue);
                }
            }
        }
        return result;
    }

    /**
     * Returns softmax of this matrix.
     *
     * @return softmax of this matrix.
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
     * @return softmax of this matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix gumbelSoftmax(Matrix result) throws MatrixException {
        return gumbelSoftmax(result, 1);
    }

    /**
     * Returns Gumbel softmax of this matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @param result result matrix.
     * @param gumbelSoftmaxTau tau value for Gumbel Softmax.
     * @return softmax of this matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix gumbelSoftmax(Matrix result, double gumbelSoftmaxTau) throws MatrixException {
        if (getColumns() != 1) {
            throw new MatrixException("Matrix must be a column vector.");
        }
        if (getRows() != result.getRows() || getColumns() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }

        double sumValue = 0;
        double epsilon = 10E-8;
        if (getMask() == null) {
            for (int row = 0; row < getRows(); row++) {
                double exp = Math.exp(getValue(row, 0));
                double value = Math.exp((Math.log(exp / (1 + exp)) + getGumbelNoise()) / gumbelSoftmaxTau);
                sumValue += value;
                result.setValue(row, 0, value);
            }
            for (int row = 0; row < getRows(); row++) {
                result.setValue(row, 0, result.getValue(row, 0) / sumValue);
            }
        }
        else {
            for (int row = 0; row < getRows(); row++) {
                if (!hasRowMaskAt(this, row) && !hasMaskAt(this, row, 0) && !hasColumnMaskAt(this, 0)) {
                    double exp = Math.exp(getValue(row, 0));
                    double value = Math.exp((Math.log(exp / (1 + exp)) + getGumbelNoise()) / gumbelSoftmaxTau);
                    sumValue += value;
                    result.setValue(row, 0, value);
                }
            }
            for (int row = 0; row < getRows(); row++) {
                if (!hasRowMaskAt(this, row) && !hasMaskAt(this, row, 0) && !hasColumnMaskAt(this, 0)) {
                    result.setValue(row, 0, result.getValue(row, 0) / sumValue);
                }
            }
        }
        return result;
    }

    /**
     * Returns Gumbel noise.<br>
     *
     * @return Gumbel noise.
     */
    private double getGumbelNoise() {
        double epsilon = 10E-8;
        return -Math.log(-Math.log(random.nextDouble() + epsilon) + epsilon);
    }

    /**
     * Returns Gumbel softmax of this matrix.<br>
     * Applies ReLU prior log function plus adds Gumbel noise.<br>
     *
     * @return Gumbel softmax of this matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix gumbelSoftmax() throws MatrixException {
        return gumbelSoftmax(new DMatrix(getRows(), getColumns()), 1);
    }

    /**
     * Returns Gumbel softmax of this matrix.<br>
     * Applies ReLU prior log function plus adds Gumbel noise.<br>
     *
     * @param gumbelSoftmaxTau tau value for Gumbel Softmax.
     * @return Gumbel softmax of this matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix gumbelSoftmax(double gumbelSoftmaxTau) throws MatrixException {
        return gumbelSoftmax(new DMatrix(getRows(), getColumns()), gumbelSoftmaxTau);
    }

    /**
     * Returns softmax gradient of this matrix.<br>
     * Assumes that input matrix is softmax result.<br>
     *
     * @param result result matrix.
     * @return softmax gradient of this matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmaxGrad(Matrix result) throws MatrixException {
        if (getColumns() != 1) {
            throw new MatrixException("Matrix must be a column vector.");
        }
        if (getRows() != result.getRows() || getRows() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }
        if (getMask() == null) {
            for (int row = 0; row < getRows(); row++) {
                for (int row1 = 0; row1 < getRows(); row1++) {
                    result.setValue(row1, row, (row == row1 ? 1 : 0) - getValue(row1, 0));
                }
            }
        }
        else {
            for (int row = 0; row < getRows(); row++) {
                if (!hasRowMaskAt(this, row) && !hasMaskAt(this, row, 0) && !hasColumnMaskAt(this, 0)) {
                    for (int row1 = 0; row1 < getRows(); row1++) {
                        result.setValue(row1, row, (row == row1 ? 1 : 0) - getValue(row1, 0));
                    }
                }
            }
        }
        return result;
    }

    /**
     * Returns softmax gradient of this matrix.
     *
     * @return softmax gradient of this matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmaxGrad() throws MatrixException {
        return softmaxGrad(new DMatrix(getRows(), getRows()));
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
     * Sets filter size for convolution operations.
     *
     * @param filterSize filter size.
     */
    public void setFilterSize(int filterSize) {
        this.filterSize = filterSize;
    }

    /**
     * Returns filter size.
     *
     * @return filter size
     */
    public int getFilterSize() {
        return filterSize;
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated value of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix convolve(Matrix filter) throws MatrixException {
        return convolve(filter, true);
    }

    /**
     * Calculates cross-correlation between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated value of cross-correlation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix crosscorrelate(Matrix filter) throws MatrixException {
        return convolve(filter, false);
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     * @return calculated value of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix convolve(Matrix filter, boolean asConvolution) throws MatrixException {
        Matrix result = new DMatrix(getRows() - filterSize + 1, getColumns() - filterSize + 1);
        convolve(filter, result, asConvolution);
        return result;
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated value of convolution.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void convolve(Matrix filter, Matrix result, boolean asConvolution) throws MatrixException {
        synchronizeProcedureFactory(filter);
        result.setProcedureFactory(procedureFactory);
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        for (int row = 0; row < result.getRows(); row += stride) {
            for (int column = 0; column < result.getColumns(); column += stride) {
                convolve(filter, result, row, column, asConvolution);
            }
        }
        if (procedureFactory != null) {
            if (asConvolution) procedureFactory.createConvolveExpression(expressionLock, this, filter, result, stride, dilation, filterSize);
            else procedureFactory.createCrosscorrelateExpression(expressionLock, this, filter, result, stride, dilation, filterSize);
        }
    }

    /**
     * Calculates convolution between this matrix and filter matrix for a specific slice row and column.
     *
     * @param filter filter
     * @param result result of operation.
     * @param rowAt position of filter.
     * @param columnAt position of filter.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     */
    private void convolve(Matrix filter, Matrix result, int rowAt, int columnAt, boolean asConvolution) {
        double resultValue = 0;
        if (getMask() == null && filter.getMask() == null) {
            for (int row = 0; row < filterSize; row++) {
                for (int column = 0; column < filterSize; column++) {
                    resultValue += getConvolutionValue(rowAt, columnAt, row, column, filterSize, filterSize, filter, asConvolution);
                }
            }
        }
        else {
            for (int row = 0; row < filterSize; row++) {
                if (!hasRowMaskAt(this, row) && !hasRowMaskAt(filter, row)) {
                    for (int column = 0; column < filterSize; column++) {
                        if (!hasMaskAt(this, row, column) && !hasColumnMaskAt(this, column) && !hasMaskAt(filter, row, column) && !hasColumnMaskAt(filter, column)) {
                            resultValue += getConvolutionValue(rowAt, columnAt, row, column, filterSize, filterSize, filter, asConvolution);
                        }
                    }
                }
            }
        }
        result.setValue(rowAt, columnAt, resultValue);
    }

    /**
     * Returns value of single convolutional operation.
     *
     * @param inputRow row at input matrix.
     * @param inputColumn column at input matrix.
     * @param sliceRow row at slice.
     * @param sliceColumn column at slice.
     * @param sliceRows number of slice rows.
     * @param sliceColumns number of slice columns.
     * @param slice reference to slice.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     * @return value of convolutional operation.
     */
    private double getConvolutionValue(int inputRow, int inputColumn, int sliceRow, int sliceColumn, int sliceRows, int sliceColumns, Matrix slice, boolean asConvolution) {
        return getValue(inputRow + sliceRow, inputColumn + sliceColumn) * getSliceValue(0, 0, sliceRow, sliceColumn, sliceRows, sliceColumns, slice, asConvolution);
    }

    /**
     * Return value at specific slice position.
     *
     * @param sliceStartRow start row at slice.
     * @param sliceStartColumn start column at slice.
     * @param sliceRow row at slice.
     * @param sliceColumn column at slice.
     * @param sliceRows number of slice rows.
     * @param sliceColumns number of slice columns.
     * @param slice reference to slice.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     * @return value of slice at specific position.
     */
    private double getSliceValue(int sliceStartRow, int sliceStartColumn, int sliceRow, int sliceColumn, int sliceRows, int sliceColumns, Matrix slice, boolean asConvolution) {
        int row = getPosition(sliceRow, sliceRows, asConvolution);
        int column = getPosition(sliceColumn, sliceColumns, asConvolution);
        return (row % dilation == 0 && column % dilation == 0) ? slice.getValue(sliceStartRow + row, sliceStartColumn + column) : 0;
    }

    /**
     * Result position at slice.
     *
     * @param pos position at slice
     * @param sliceSize size of slice.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     * @return position.
     */
    private int getPosition(int pos, int sliceSize, boolean asConvolution) {
        return asConvolution ? sliceSize - 1 - pos : pos;
    }

    /**
     * Calculates gradient of convolution for output.
     *
     * @param filter filter for convolutional operator.
     * @return output gradient.
     */
    public Matrix convolveOutputGradient(Matrix filter) {
        return convolveOutputGradient(filter, true);
    }

    /**
     * Calculates gradient of cross-correlation for output.
     *
     * @param filter filter for cross-correlation operator.
     * @return output gradient.
     */
    public Matrix crosscorrelateOutputGradient(Matrix filter) {
        return convolveOutputGradient(filter, false);
    }

    /**
     * Calculates gradient of convolution for output.
     *
     * @param filter filter for convolutional operator.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     * @return output gradient.
     */
    public Matrix convolveOutputGradient(Matrix filter, boolean asConvolution) {
        Matrix result = new DMatrix(getRows() + filterSize - 1, getColumns() + filterSize - 1);
        convolveOutputGradient(filter, result, asConvolution);
        return result;
    }

    /**
     * Calculates gradient of convolution for output.
     *
     * @param filter filter for convolutional operator.
     * @param result output gradient.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     */
    public void convolveOutputGradient(Matrix filter, Matrix result, boolean asConvolution) {
        for (int row = 0; row < getRows(); row += stride) {
            for (int column = 0; column < getColumns(); column += stride) {
                convolveGradient(filter, result, row, column, true, filterSize, filterSize, asConvolution);
            }
        }
    }

    /**
     * Calculates gradient of convolution for filter.
     *
     * @param input input for convolutional operator.
     * @return filter gradient.
     */
    public Matrix convolveFilterGradient(Matrix input) {
        return convolveFilterGradient(input, true);
    }

    /**
     * Calculates gradient of cross-correlation for filter.
     *
     * @param input input for cross-correlation operator.
     * @return filter gradient.
     */
    public Matrix crosscorrelateFilterGradient(Matrix input) {
        return convolveFilterGradient(input, false);
    }

    /**
     * Calculates gradient of convolution for filter.
     *
     * @param input input for convolutional operator.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     * @return filter gradient.
     */
    public Matrix convolveFilterGradient(Matrix input, boolean asConvolution) {
        Matrix result = new DMatrix(input.getRows() - getRows() + 1, input.getColumns() - getColumns() + 1);
        convolveFilterGradient(input, result, asConvolution);
        return result;
    }

    /**
     * Calculates gradient of convolution for filter.
     *
     * @param input input for convolutional operator.
     * @param resultGradient filter gradient.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     */
    public void convolveFilterGradient(Matrix input, Matrix resultGradient, boolean asConvolution) {
        for (int row = 0; row < getRows(); row += stride) {
            for (int column = 0; column < getColumns(); column += stride) {
                convolveGradient(input, resultGradient, row, column, false, resultGradient.getRows(), resultGradient.getColumns(), asConvolution);
            }
        }
    }

    /**
     * Calculates gradient slice for convolution operation. This matrix is output gradient for calculation.
     *
     * @param input input matrix for calculation.
     * @param result resulting gradient matrix.
     * @param gradientRowAt gradient row.
     * @param gradientColumnAt gradient column.
     * @param slideResult if true slides over result otherwise slides over input.
     * @param sliceRows size of slice (filter) in rows.
     * @param sliceColumns size of slice (filter) in columns.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     */
    private void convolveGradient(Matrix input, Matrix result, int gradientRowAt, int gradientColumnAt, boolean slideResult, int sliceRows, int sliceColumns, boolean asConvolution) {
        int inputRowAt = slideResult ? 0 : gradientRowAt;
        int inputColumnAt = slideResult ? 0 : gradientColumnAt;
        int resultGradientRowAt = slideResult ? gradientRowAt : 0;
        int resultGradientColumnAt = slideResult ? gradientColumnAt : 0;
        double gradientValue = getValue(gradientRowAt, gradientColumnAt);
        if (getMask() == null && input.getMask() == null) {
            for (int row = 0; row < sliceRows; row++) {
                for (int column = 0; column < sliceColumns; column++) {
                    updateConvolutionGradientValue(resultGradientRowAt, resultGradientColumnAt, inputRowAt, inputColumnAt, row, column, sliceRows, sliceColumns, gradientValue, input, result, asConvolution);
                }
            }
        }
        else {
            for (int row = 0; row < sliceRows; row++) {
                if (!hasRowMaskAt(this, row) && !hasRowMaskAt(input, sliceRows - 1 - row)) {
                    for (int column = 0; column < sliceColumns; column++) {
                        if (!hasMaskAt(this, row, column) && !hasColumnMaskAt(this, column) && !hasMaskAt(input, sliceRows - 1 - row, column) && !hasColumnMaskAt(input, sliceColumns - 1 - column)) {
                            updateConvolutionGradientValue(resultGradientRowAt, resultGradientColumnAt, inputRowAt, inputColumnAt, row, column, sliceRows, sliceColumns, gradientValue, input, result, asConvolution);
                        }
                    }
                }
            }
        }
    }

    /**
     * Updates convolution gradient value.
     *
     * @param resultStartRow result start row.
     * @param resultStartColumn result start column.
     * @param sliceStartRow start row at slice.
     * @param sliceStartColumn start column at slice.
     * @param sliceRow row at slice.
     * @param sliceColumn column at slice.
     * @param sliceRows number of slice rows.
     * @param sliceColumns number of slice columns.
     * @param gradientValue gradient value.
     * @param input input matrix.
     * @param result result matrix.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     */
    private void updateConvolutionGradientValue(int resultStartRow, int resultStartColumn, int sliceStartRow, int sliceStartColumn, int sliceRow, int sliceColumn, int sliceRows, int sliceColumns, double gradientValue, Matrix input, Matrix result, boolean asConvolution) {
        result.incrementByValue(resultStartRow + sliceRow, resultStartColumn + sliceColumn, getSliceValue(sliceStartRow, sliceStartColumn, sliceRow, sliceColumn, sliceRows, sliceColumns, input, asConvolution) * gradientValue);
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
     * @param maxArgumentsAt arguments on maximum row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix maxPool(int [][][] maxArgumentsAt) throws MatrixException {
        Matrix result = new DMatrix(getRows() - poolSize + 1, getColumns() - poolSize + 1);
        maxPool(result, maxArgumentsAt);
        return result;
    }

    /**
     * Calculates max pooling operation for this matrix and returns max arguments.
     *
     * @param result result matrix.
     * @param maxArgumentsAt arguments on maximum row and col value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void maxPool(Matrix result, int [][][] maxArgumentsAt) throws MatrixException {
        result.setProcedureFactory(procedureFactory);
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        for (int row = 0; row < result.getRows(); row += stride) {
            for (int column = 0; column < result.getColumns(); column += stride) {
                maxPool(result, row, column, maxArgumentsAt);
            }
        }
        if (procedureFactory != null) procedureFactory.createMaxPoolExpression(expressionLock, this, result, stride, poolSize);
    }

    /**
     * Calculates max pooling operation for this matrix and returns max arguments.
     *
     * @param result result matrix.
     * @param maxArgumentsAt arguments on maximum row and col value.
     */
    private void maxPool(Matrix result, int rowAt, int columnAt, int [][][] maxArgumentsAt) {
        double maxValue = Double.NEGATIVE_INFINITY;
        if (getMask() == null) {
            for (int row = 0; row < poolSize; row++) {
                for (int column = 0; column < poolSize; column++) {
                    int currentRow = rowAt + row;
                    int currentColumn = columnAt + column;
                    double currentValue = getValue(currentRow, currentColumn);
                    if (maxValue < currentValue) {
                        maxValue = currentValue;
                        maxArgumentsAt[rowAt][columnAt][0] = currentRow;
                        maxArgumentsAt[rowAt][columnAt][1] = currentColumn;
                    }
                }
            }
        }
        else {
            for (int row = 0; row < poolSize; row++) {
                if (!hasRowMaskAt(this, row)) {
                    for (int column = 0; column < poolSize; column++) {
                        if (!hasMaskAt(this, row, column) && !hasColumnMaskAt(this, column)) {
                            int currentRow = rowAt + row;
                            int currentColumn = columnAt + column;
                            double currentValue = getValue(currentRow, currentColumn);
                            if (maxValue < currentValue) {
                                maxValue = currentValue;
                                maxArgumentsAt[rowAt][columnAt][0] = currentRow;
                                maxArgumentsAt[rowAt][columnAt][1] = currentColumn;
                            }
                        }
                    }
                }
            }
        }
        result.setValue(rowAt, columnAt, maxValue);
    }

    /**
     * Calculates gradient of max pooling operation for this matrix.
     *
     * @param maxArgumentsAt arguments on maximum row and col value.
     * @return result matrix.
     */
    public Matrix maxPoolGradient(int [][][] maxArgumentsAt) {
        Matrix result = new DMatrix(getRows() + poolSize - 1, getColumns() + poolSize - 1);
        maxPoolGradient(result, maxArgumentsAt);
        return result;
    }

    /**
     * Calculates gradient for max pool operation.
     *
     * @param result result matrix.
     * @param maxArgumentsAt arguments on maximum row and col value.
     */
    public void maxPoolGradient(Matrix result, int[][][] maxArgumentsAt) {
        for (int row = 0; row < getRows(); row += stride) {
            for (int column = 0; column < getColumns(); column += stride) {
                maxPoolGradient(result, row, column, maxArgumentsAt[row][column]);
            }
        }
    }

    /**
     * Calculates gradient for max pool operation at certain row and column position.
     *
     * @param result result matrix.
     * @param rowAt row position.
     * @param columnAt column position.
     * @param maxArgumentsAt arguments on maximum row and col value.
     */
    private void maxPoolGradient(Matrix result, int rowAt, int columnAt, int[] maxArgumentsAt) {
        if (getMask() == null) result.setValue(maxArgumentsAt[0], maxArgumentsAt[1], getValue(rowAt, columnAt));
        else {
            if (!hasRowMaskAt(this, rowAt) && !hasMaskAt(this, rowAt, columnAt) && !hasColumnMaskAt(this, columnAt)) {
                result.setValue(maxArgumentsAt[0], maxArgumentsAt[1],  getValue(rowAt, columnAt));
            }
        }
    }

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix averagePool() throws MatrixException {
        Matrix result = new DMatrix(getRows() - poolSize + 1, getColumns() - poolSize + 1);
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
        result.setProcedureFactory(procedureFactory);
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        for (int row = 0; row < result.getRows(); row += stride) {
            for (int column = 0; column < result.getColumns(); column += stride) {
                averagePool(result, row, column);
            }
        }
        if (procedureFactory != null) procedureFactory.createAveragePoolExpression(expressionLock, this, result, stride, poolSize);
    }

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @param result result matrix.
     */
    private void averagePool(Matrix result, int rowAt, int columnAt) {
        double sumValue = 0;
        if (getMask() == null) {
            for (int row = 0; row < poolSize; row++) {
                for (int column = 0; column < poolSize; column++) {
                    sumValue += getValue(rowAt + row, columnAt + column);
                }
            }
        }
        else {
            for (int row = 0; row < poolSize; row++) {
                if (!hasRowMaskAt(this, row)) {
                    for (int column = 0; column < poolSize; column++) {
                        if (!hasMaskAt(this, row, column) && !hasColumnMaskAt(this, column)) {
                            sumValue += getValue(rowAt + row, columnAt + column);
                        }
                    }
                }
            }
        }
        result.setValue(rowAt, columnAt, sumValue / (double)(poolSize * poolSize));
    }

    /**
     * Calculates gradient of average pooling operation for this matrix.
     *
     * @return result matrix.
     */
    public Matrix averagePoolGradient() {
        Matrix result = new DMatrix(getRows() + poolSize - 1, getColumns() + poolSize - 1);
        averagePoolGradient(result);
        return result;
    }

    /**
     * Calculates gradient of average pooling operation for this matrix.
     *
     * @param result result matrix.
     */
    public void averagePoolGradient(Matrix result) {
        for (int row = 0; row < result.getRows(); row += stride) {
            for (int column = 0; column < result.getColumns(); column += stride) {
                averagePoolGradient(result, row, column);
            }
        }
    }

    /**
     * Calculates gradient of average pooling operation for this matrix.
     *
     * @param result result matrix.
     */
    private void averagePoolGradient(Matrix result, int rowAt, int columnAt) {
        if (getMask() == null) result.setValue(rowAt, columnAt, 1 / (double)(poolSize * poolSize));
        else {
            if (!hasRowMaskAt(this, rowAt) && !hasMaskAt(this, rowAt, columnAt) && !hasColumnMaskAt(this, columnAt)) {
                result.setValue(rowAt, columnAt, 1 / (double)(poolSize * poolSize));
            }
        }
    }

    /**
     * Transposes matrix.
     *
     * @return reference to this matrix but with transposed that is flipped rows and columns.
     * @throws MatrixException throws exception if mask operation fails.
     */
    public Matrix transpose() throws MatrixException {
        try {
            // Make shallow copy of matrix leaving references internal objects which are shared.
            Matrix clone = (Matrix)clone();
            clone.isTransposed = !clone.isTransposed; // transpose
            if (getMask() != null) clone.setMask(getMask().transpose());
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
    public boolean isTransposed() {
        return isTransposed;
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
     * Returns if matrix has mask at specific position.
     *
     * @param matrix matrix as input.
     * @param row specific row.
     * @param column specific column.
     * @return if true mask exists and is masked at specific position (row + column).
     */
    public static boolean hasMaskAt(Matrix matrix, int row, int column) {
        return matrix.getMask() != null && matrix.getMask().getMask(row, column);
    }

    /**
     * Returns if matrix has mask at specific row.
     *
     * @param matrix matrix as input.
     * @param row specific row.
     * @return if true mask exists and is masked at specific row.
     */
    public static boolean hasRowMaskAt(Matrix matrix, int row) {
        return matrix.getMask() == null || matrix.getMask().getRowMask(row);
    }

    /**
     * Returns if matrix has mask at specific column.
     *
     * @param matrix matrix as input.
     * @param column specific column.
     * @return if true mask exists and is masked at specific column.
     */
    public static boolean hasColumnMaskAt(Matrix matrix, int column) {
        return matrix.getMask() == null || matrix.getMask().getColumnMask(column);
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
