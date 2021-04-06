package utils.matrix;

import utils.DynamicParamException;
import utils.procedure.ProcedureFactory;

import java.io.Serializable;
import java.util.Random;

/**
 * Abstract class that implements common operations for matrices.
 *
 */
public abstract class AbstractMatrix implements Cloneable, Serializable, Matrix {

    private static final long serialVersionUID = 4372639167186260605L;

    /**
     * Initializer variable.
     *
     */
    private Matrix.Initializer initializer;

    /**
     * Reference to mask of matrix. If null mask is not used.
     *
     */
    private Mask mask;

    /**
     * If true matrix is treated as scalar (1x1) matrix otherwise as normal matrix.
     *
     */
    private final boolean isScalar;

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
    private String name;

    /**
     * Constructor for matrix.
     *
     * @param isScalar true if matrix is scalar (size 1x1).
     */
    protected AbstractMatrix(boolean isScalar) {
        this.isScalar = isScalar;
    }

    /**
     * Constructor for matrix.
     *
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param name name if matrix.
     */
    protected AbstractMatrix(boolean isScalar, String name) {
        this(isScalar);
        this.name = name;
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
     * Sets initializer of matrix.
     *
     * @param initializer initializer.
     */
    public void setInitializer(Matrix.Initializer initializer) {
        this.initializer = initializer;
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
            case ZERO:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> 0;
                break;
            case ONE:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> 1;
                initialize(initializer);
                break;
            case RANDOM:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> random.nextDouble();
                initialize(initializer);
                break;
            case IDENTITY:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> (row == col) ? 1 : 0;
                initialize(initializer);
                break;
            case NORMAL_XAVIER:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / (double)(getRows() + getColumns())));
                initialize(initializer);
                break;
            case UNIFORM_XAVIER:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(getRows() + getColumns())));
                initialize(initializer);
                break;
            case NORMAL_HE:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / ((double)getRows())));
                initialize(initializer);
                break;
            case UNIFORM_HE:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(getRows())));
                initialize(initializer);
                break;
            case NORMAL_LECUN:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> normal(Math.sqrt(1 / (double)(getRows())));
                initialize(initializer);
                break;
            case UNIFORM_LECUN:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> uniform(Math.sqrt(3 / (double)(getRows())));
                initialize(initializer);
                break;
            case NORMAL_XAVIER_CONV:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / (double)(outputs + inputs)));
                initialize(initializer);
                break;
            case UNIFORM_XAVIER_CONV:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(outputs + inputs)));
                initialize(initializer);
                break;
            case NORMAL_HE_CONV:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / (double)(outputs)));
                initialize(initializer);
                break;
            case UNIFORM_HE_CONV:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(outputs)));
                initialize(initializer);
                break;
            case NORMAL_LECUN_CONV:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> normal(Math.sqrt(1 / (double)(outputs)));
                initialize(initializer);
                break;
            case UNIFORM_LECUN_CONV:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> uniform(Math.sqrt(3 / (double)(outputs)));
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
     * Initializes matrix with given initializer operation.
     *
     * @param initializer initializer operation.
     */
    public void initialize(Matrix.Initializer initializer) {
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
            newMatrix.setInitializer(initializer);
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
            newMatrix.setInitializer(initializer);
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
     * Returns placeholder for result matrix.
     *
     * @param other other matrix.
     * @return result matrix placeholder.
     */
    private Matrix getResultMatrix(Matrix other) {
        return !isScalar() ? getNewMatrix() : other.getNewMatrix();
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
    public Matrix apply(Matrix.MatrixUnaryOperation matrixUnaryOperation) throws MatrixException {
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
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        applyDot(other, result);
        if (procedureFactory != null) procedureFactory.createDotExpression(expressionLock, this, other, result);
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
     * Returns constant as constant matrix.
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
     * @return cumulative sum of this matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix sumAsMatrix() throws MatrixException {
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        Matrix result = constantAsMatrix(sum());
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) procedureFactory.createSumExpression(expressionLock, this, result);
        return result;
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
        Matrix result = constantAsMatrix(mean());
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
        Matrix result = constantAsMatrix(variance());
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
    public Matrix varianceAsMatrix(Matrix mean) {
        return constantAsMatrix(variance(mean.getValue(0, 0)));
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
    public Matrix standardDeviationAsMatrix(Matrix mean) {
        return constantAsMatrix(standardDeviation(mean.getValue(0, 0)));
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
        Matrix result = constantAsMatrix(norm(p));
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) procedureFactory.createNormExpression(expressionLock, this, result, p);
        return result;
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
    public Matrix minAsMatrix() {
        return constantAsMatrix(min());
    }

    /**
     * Returns maximum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return maximum value of matrix.
     */
    public Matrix maxAsMatrix() {
        return constantAsMatrix(max());
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
     * @return Gumbel softmax of this matrix.
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
     * @return Gumbel softmax of this matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix gumbelSoftmax(double gumbelSoftmaxTau) throws MatrixException {
        return gumbelSoftmax(new DMatrix(getRows(), getColumns()), gumbelSoftmaxTau);
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
        Matrix result = new DMatrix(getRows() - getFilterSize() + 1, getColumns() - getFilterSize() + 1);
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
        applyConvolve(filter, result, asConvolution);
        if (procedureFactory != null) {
            if (asConvolution) procedureFactory.createConvolveExpression(expressionLock, this, filter, result, getStride(), getDilation(), getFilterSize());
            else procedureFactory.createCrosscorrelateExpression(expressionLock, this, filter, result, getStride(), getDilation(), getFilterSize());
        }
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated value of convolution.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     */
    protected abstract void applyConvolve(Matrix filter, Matrix result, boolean asConvolution);

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
        Matrix result = new DMatrix(getRows() + getFilterSize() - 1, getColumns() + getFilterSize() - 1);
        convolveOutputGradient(filter, result, asConvolution);
        return result;
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
     * Calculates max pooling operation for this matrix and returns max arguments.
     *
     * @param maxArgumentsAt arguments on maximum row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix maxPool(int [][][] maxArgumentsAt) throws MatrixException {
        Matrix result = new DMatrix(getRows() - getPoolSize() + 1, getColumns() - getPoolSize() + 1);
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
        applyMaxPool(result, maxArgumentsAt);
        if (procedureFactory != null) procedureFactory.createMaxPoolExpression(expressionLock, this, result, getStride(), getPoolSize());
    }

    /**
     * Calculates max pooling operation for this matrix and returns max arguments.
     *
     * @param result result matrix.
     * @param maxArgumentsAt arguments on maximum row and col value.
     */
    protected abstract void applyMaxPool(Matrix result, int [][][] maxArgumentsAt);

    /**
     * Calculates gradient of max pooling operation for this matrix.
     *
     * @param maxArgumentsAt arguments on maximum row and col value.
     * @return result matrix.
     */
    public Matrix maxPoolGradient(int [][][] maxArgumentsAt) {
        Matrix result = new DMatrix(getRows() + getPoolSize() - 1, getColumns() + getPoolSize() - 1);
        maxPoolGradient(result, maxArgumentsAt);
        return result;
    }

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix averagePool() throws MatrixException {
        Matrix result = new DMatrix(getRows() - getPoolSize() + 1, getColumns() - getPoolSize() + 1);
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
        applyAveragePool(result);
        if (procedureFactory != null) procedureFactory.createAveragePoolExpression(expressionLock, this, result, getStride(), getPoolSize());
    }

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @param result result matrix.
     */
    protected abstract void applyAveragePool(Matrix result);

    /**
     * Calculates gradient of average pooling operation for this matrix.
     *
     * @return result matrix.
     */
    public Matrix averagePoolGradient() {
        Matrix result = new DMatrix(getRows() + getPoolSize() - 1, getColumns() + getPoolSize() - 1);
        averagePoolGradient(result);
        return result;
    }

    /**
     * Transposes matrix.
     *
     * @return new matrix but as transposed that is with flipped rows and columns.
     * @throws MatrixException throws exception if transpose operation fails.
     */
    public Matrix transpose() throws MatrixException {
        Matrix transposedMatrix = getNewMatrix(true);
        for (int row = 0; row < getRows(); row++) {
            for (int column = 0; column < getColumns(); column++) {
                transposedMatrix.setValue(column, row, getValue(row, column));
            }
        }
        if (getMask() != null) transposedMatrix.setMask(getMask().transpose());
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
     * Returns if matrix has mask at specific position.
     *
     * @param matrix matrix as input.
     * @param row specific row.
     * @param column specific column.
     * @return if true mask exists and is masked at specific position (row + column).
     */
    public boolean hasMaskAt(Matrix matrix, int row, int column) {
        return matrix.getMask() != null && matrix.getMask().getMask(row, column);
    }

    /**
     * Returns if matrix has mask at specific row.
     *
     * @param matrix matrix as input.
     * @param row specific row.
     * @return if true mask exists and is masked at specific row.
     */
    public boolean hasRowMaskAt(Matrix matrix, int row) {
        return matrix.getMask() == null || matrix.getMask().getRowMask(row);
    }

    /**
     * Returns if matrix has mask at specific column.
     *
     * @param matrix matrix as input.
     * @param column specific column.
     * @return if true mask exists and is masked at specific column.
     */
    public boolean hasColumnMaskAt(Matrix matrix, int column) {
        return matrix.getMask() == null || matrix.getMask().getColumnMask(column);
    }

    /**
     * Returns new mask for this matrix.<br>
     * Implemented by underlying matrix class.<br>
     *
     * @return mask of this matrix.
     */
    protected abstract Mask getNewMask();

}
