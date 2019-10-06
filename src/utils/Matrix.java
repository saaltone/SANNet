/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package utils;

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
         * @param col col of matrix if relevant for initialization.
         * @return value to be used for initialization.
         */
        double value(int row, int col);
    }

    /**
     * Defines interface to be used as part of lambda function to execute two argument matrix operation.
     */
    public interface MatrixBiOperation {
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
    public interface MatrixUniOperation {
        /**
         * Defines operation to be executed with single parameter.
         *
         * @param value1 value for parameter.
         * @return value returned by the operation.
         */
        double execute(double value1);
    }

    /**
     * Defines if Matrix is transposed (true) or not (false).
     */
    protected boolean t;

    /**
     * Initializer variable.
     */
    protected Initializer initializer;

    /**
     * Defines if matrix is masked (true) or not (false).<br>
     * Masking provides utility to ignore specific matrix entries (row, column) when executing matrix operations.<br>
     */
    protected boolean masked = false;

    /**
     * Bernoulli probability for selecting if matrix entry (row, column) is masked or not.
     */
    protected double proba = 0;

    /**
     * Scaling constant applied in all matrix operations meaning scalingConstant * operation.
     */
    protected double scalingConstant = 1;

    /**
     * Row where slice starts at. Used or crosscorrelation and convolutional operators.
     *
     */
    private int sliceAtRow = 0;

    /**
     * Column where slice starts at. Used or crosscorrelation and convolutional operators.
     *
     */
    private int sliceAtCol = 0;

    /**
     * Size of slice in rows. Used or crosscorrelation and convolutional operators.
     *
     */
    private int sliceRowSize = 0;

    /**
     * Size of slice in columnss. Used or crosscorrelation and convolutional operators.
     *
     */
    private int sliceColSize = 0;

    /**
     * Autogradient for matrix.
     *
     */
    protected transient ProcedureFactory procedureFactory = null;

    /**
     * Random function for matrix class.
     */
    protected final Random random = new Random();

    /**
     * Default constructor for matrix class
     */
    public Matrix() {
    }

    public void initializeSlice() {
        sliceRowSize = getRows();
        sliceColSize = getCols();
    }

    /**
     * Function used to reinitialize matrix and it's mask.
     */
    public void reset() {
        resetMatrix();
        resetMask();
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
    protected double uniform(double range) {
        return (2 * random.nextDouble()- 1)  * range;
    }

    /**
     * Returns value from normal distribution defined by standard deviation (stdev).
     *
     * @param stdev standard deviation of the distribution.
     * @return random value drawn from the distribution.
     */
    protected double normal(double stdev) {
        return random.nextGaussian() * stdev;
    }

    /**
     * Default initialization without parameter.
     *
     * @param initialization type of initialization defined in class Init.
     */
    public void init(Init initialization) {
        init(initialization, 0, 0);
    }

    /**
     * Default initialization with parameter.
     *
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     */
    public void init(Init initialization, int inputs, int outputs) {
        switch (initialization) {
            case ZERO:
                initializer = (Initializer & Serializable) (row, col) -> 0;
                break;
            case ONE:
                initializer = (Initializer & Serializable) (row, col) -> 1;
                break;
            case RANDOM:
                initializer = (Initializer & Serializable) (row, col) -> random.nextDouble();
                break;
            case IDENTITY:
                initializer = (Initializer & Serializable) (row, col) -> (row == col) ? 1 : 0;
                break;
            case NORMAL_XAVIER:
                initializer = (Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / (double)(getRows() + getCols())));
                break;
            case UNIFORM_XAVIER:
                initializer = (Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(getRows() + getCols())));
                break;
            case NORMAL_HE:
                initializer = (Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / ((double)getRows())));
                break;
            case UNIFORM_HE:
                initializer = (Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(getRows())));
                break;
            case NORMAL_LECUN:
                initializer = (Initializer & Serializable) (row, col) -> normal(Math.sqrt(1 / (double)(getRows())));
                break;
            case UNIFORM_LECUN:
                initializer = (Initializer & Serializable) (row, col) -> uniform(Math.sqrt(3 / (double)(getRows())));
                break;
            case NORMAL_XAVIER_CONV:
                initializer = (Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / (double)(outputs + inputs)));
                break;
            case UNIFORM_XAVIER_CONV:
                initializer = (Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(outputs + inputs)));
                break;
            case NORMAL_HE_CONV:
                initializer = (Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / (double)(outputs)));
                break;
            case UNIFORM_HE_CONV:
                initializer = (Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(outputs)));
                break;
            case NORMAL_LECUN_CONV:
                initializer = (Initializer & Serializable) (row, col) -> normal(Math.sqrt(1 / (double)(outputs)));
                break;
            case UNIFORM_LECUN_CONV:
                initializer = (Initializer & Serializable) (row, col) -> uniform(Math.sqrt(3 / (double)(outputs)));
                break;
            default:
                break;
        }
        initialize(initializer);
    }

    /**
     * Returns matrix initializer.
     *
     * @return current matrix initializer instance.
     */
    public Initializer getInit() {
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
     * Gets size (rows * columns) of matrix.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @return size of matrix.
     */
    public abstract int getSize();

    /**
     * Gets number of rows in matrix.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @return number of rows in matrix.
     */
    public abstract int getRows();

    /**
     * Gets number of columns in matrix.<br>
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
     */
    public Matrix reference() {
        Matrix newMatrix = null;
        // Make shallow copy of matrix leaving references internal objects which are shared.
        try {
            newMatrix = (Matrix)super.clone();
            newMatrix.initializer = initializer;
        } catch (CloneNotSupportedException exception) {}
        return newMatrix;
    }

    /**
     * Creates new matrix with object full copy of this matrix.
     *
     * @return newly created reference matrix.
     */
    public Matrix copy() {
        Matrix newMatrix = null;
        // Make shallow copy of matrix leaving references internal objects which are shared.
        try {
            newMatrix = (Matrix)super.clone();
            newMatrix.initializer = initializer;
            // Copy matrix data
            newMatrix.setAsMatrix(this);
        } catch (CloneNotSupportedException exception) {}
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
     * Adds uni argument function into procedure factory and propagates procedure factory.
     *
     * @param result result matrix
     * @param uniFunction uniFunction to be applied
     */
    private void addToProcedureFactory(Matrix result, UniFunction uniFunction) throws MatrixException {
        if (procedureFactory != null) {
            procedureFactory.addExpression(this, result, uniFunction);
            result.setProcedureFactory(procedureFactory);
        }
    }

    /**
     * Adds bi argument function into procedure factory and propagates procedure factory.
     *
     * @param other other matrix
     * @param result result matrix
     * @param biFunction function to be applied
     */
    private void addToProcedureFactory(Matrix other, Matrix result, BiFunction biFunction) throws MatrixException {
        ProcedureFactory otherProcedureFactory = other.getProcedureFactory();
        if (procedureFactory == null && otherProcedureFactory == null) return;
        if (procedureFactory != null && otherProcedureFactory != null && procedureFactory != otherProcedureFactory) throw new MatrixException("This and other matrices have conflicting procedure factories.");
        ProcedureFactory currentProcedureFactory = procedureFactory != null ? procedureFactory : otherProcedureFactory;
        currentProcedureFactory.addExpression(this, other, result, biFunction);
        result.setProcedureFactory(currentProcedureFactory);
    }

    /**
     * Adds bi argument expression into procedure factory and propagates procedure factory.
     *
     * @param other other matrix
     * @param result result matrix
     * @param type type of expression
     */
    private void addToProcedureFactory(Matrix other, Matrix result, Expression.Type type) throws MatrixException {
        ProcedureFactory otherProcedureFactory = other.getProcedureFactory();
        if (procedureFactory == null && otherProcedureFactory == null) return;
        if (procedureFactory != null && otherProcedureFactory != null && procedureFactory != otherProcedureFactory) throw new MatrixException("This and other matrices have conflicting procedure factories.");
        ProcedureFactory currentProcedureFactory = procedureFactory != null ? procedureFactory : otherProcedureFactory;
        currentProcedureFactory.addExpression(this, other, result, type);
        result.setProcedureFactory(currentProcedureFactory);
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
                if (isMasked()) setMask(row, col, other.getMask(row, col));
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
    public Matrix apply(Matrix result, MatrixUniOperation operation) throws MatrixException {
        if (result.getRows() != getRows() || result.getCols() != getCols()) {
            throw new MatrixException("Incompatible result matrix sizes: " + result.getRows() + "x" + result.getCols());
        }
        if (!masked) {
            for (int row = 0; row < getRows(); row++) {
                for (int col = 0; col < getCols(); col++) {
                    result.setValue(row, col, scalingConstant * operation.execute(getValue(row, col)));
                }
            }
        }
        else {
            for (int row = 0; row < getRows(); row++) {
                if (!getRowMask(row)) {
                    for (int col = 0; col < getCols(); col++) {
                        if (!getMask(row, col) && !getColMask(col)) {
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
    public Matrix apply(MatrixUniOperation operation) throws MatrixException {
        return apply(getNewMatrix(getRows(), getCols()), operation);
    }

    /**
     * Applies uniFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param uniFunctionType uniFunction type to be applied.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(Matrix result, UniFunctionType uniFunctionType) throws MatrixException {
        UniFunction uniFunction = new UniFunction(uniFunctionType);
        apply(result, uniFunction.getFunction());
        addToProcedureFactory(result, uniFunction);
    }

    /**
     * Applies uniFunction to this matrix and return operation result.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param uniFunctionType uniFunction type to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix apply(UniFunctionType uniFunctionType) throws MatrixException {
        UniFunction uniFunction = new UniFunction(uniFunctionType);
        Matrix result = apply(getNewMatrix(getRows(), getCols()), uniFunction.getFunction());
        addToProcedureFactory(result, uniFunction);
        return result;
    }

    /**
     * Applies uniFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param uniFunction uniFunction to be applied.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(Matrix result, UniFunction uniFunction) throws MatrixException {
        apply(result, uniFunction.getFunction());
        addToProcedureFactory(result, uniFunction);
    }

    /**
     * Applies uniFunction to this matrix and return operation result.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param uniFunction uniFunction to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix apply(UniFunction uniFunction) throws MatrixException {
        Matrix result = apply(getNewMatrix(getRows(), getCols()), uniFunction.getFunction());
        addToProcedureFactory(result, uniFunction);
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
    public Matrix applyBi(Matrix other, Matrix result, Matrix.MatrixBiOperation operation) throws MatrixException {
        if (getRows() != other.getRows() || getCols() != other.getCols()) {
            throw new MatrixException("Incompatible matrix sizes: " + getRows() + "x" + getCols() + " by " + other.getRows() + "x" + other.getCols());
        }
        if (getRows() != result.getRows() || getCols() != result.getCols()) {
            throw new MatrixException("Incompatible result matrix sizes: " + result.getRows() + "x" + result.getCols());
        }
        if (!masked || other.isMasked()) {
            for (int row = 0; row < getRows(); row++) {
                for (int col = 0; col < getCols(); col++) {
                    result.setValue(row, col, scalingConstant * operation.execute(getValue(row, col), other.getValue(row, col)));
                }
            }
        }
        else {
            for (int row = 0; row < getRows(); row++) {
                if (!getRowMask(row) && !other.getRowMask(row)) {
                    for (int col = 0; col < getCols(); col++) {
                        if (!getMask(row, col) && !other.getMask(row, col) && !getColMask(col) && !other.getColMask(col)) {
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
    public Matrix applyBi(Matrix other, Matrix.MatrixBiOperation operation) throws MatrixException {
        return applyBi(other, getNewMatrix(getRows(), getCols()), operation);
    }

    /**
     * Applies biFunction to this matrix.<br>
     * Example of operation can be applying power operation to this and other matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param other other matrix
     * @param result result matrix.
     * @param biFunctionType biFunction type to be applied.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyBi(Matrix other, Matrix result, BiFunctionType biFunctionType) throws MatrixException {
        BiFunction biFunction = new BiFunction(biFunctionType);
        applyBi(other, result, biFunction.getFunction());
        addToProcedureFactory(other, result, biFunction);
    }

    /**
     * Applies biFunction to this matrix and return operation result.<br>
     * Example of operation can be applying power operation to this and other matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param other other matrix
     * @param biFunctionType biFunction type to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix applyBi(Matrix other, BiFunctionType biFunctionType) throws MatrixException {
        BiFunction biFunction = new BiFunction(biFunctionType);
        Matrix result = applyBi(other, getNewMatrix(getRows(), getCols()), biFunction.getFunction());
        addToProcedureFactory(other, result, biFunction);
        return result;
    }

    /**
     * Applies biFunction to this matrix.<br>
     * Example of operation can be applying power operation to this and other matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param other other matrix
     * @param result result matrix.
     * @param biFunction biFunction to be applied.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyBi(Matrix other, Matrix result, BiFunction biFunction) throws MatrixException {
        applyBi(other, result, biFunction.getFunction());
        addToProcedureFactory(other, result, biFunction);
    }

    /**
     * Applies biFunction to this matrix and return operation result.<br>
     * Example of operation can be applying power operation to this and other matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param other other matrix
     * @param biFunction biFunction to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public Matrix applyBi(Matrix other, BiFunction biFunction) throws MatrixException {
        Matrix result = applyBi(other, getNewMatrix(getRows(), getCols()), biFunction.getFunction());
        addToProcedureFactory(other, result, biFunction);
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
        applyBi (other, result, (Matrix.MatrixBiOperation & Serializable) Double::sum);
        addToProcedureFactory(other, result, Expression.Type.ADD);
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
        Matrix result = applyBi (other, (Matrix.MatrixBiOperation & Serializable) Double::sum);
        addToProcedureFactory(other, result, Expression.Type.ADD);
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
        addToProcedureFactory(constantMatrix, result, Expression.Type.ADD);
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
        addToProcedureFactory(constantMatrix, result, Expression.Type.ADD);
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
        applyBi (other, result, (Matrix.MatrixBiOperation & Serializable) (value1, value2) -> value1 - value2);
        addToProcedureFactory(other, result, Expression.Type.SUB);
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
        Matrix result = applyBi (other, (Matrix.MatrixBiOperation & Serializable) (value1, value2) -> value1 - value2);
        addToProcedureFactory(other, result, Expression.Type.SUB);
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
        addToProcedureFactory(constantMatrix, result, Expression.Type.SUB);
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
        addToProcedureFactory(constantMatrix, result, Expression.Type.SUB);
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
        applyBi (other, result, (Matrix.MatrixBiOperation & Serializable) (value1, value2) -> value1 * value2);
        addToProcedureFactory(other, result, Expression.Type.MUL);
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
        Matrix result = applyBi (other, (Matrix.MatrixBiOperation & Serializable) (value1, value2) -> value1 * value2);
        addToProcedureFactory(other, result, Expression.Type.MUL);
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
        addToProcedureFactory(constantMatrix, result, Expression.Type.MUL);
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
        addToProcedureFactory(constantMatrix, result, Expression.Type.MUL);
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
        applyBi (other, result, (Matrix.MatrixBiOperation & Serializable) (value1, value2) -> value2 != 0 ? value1 / value2 : Double.MAX_VALUE);
        addToProcedureFactory(other, result, Expression.Type.DIV);
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
        Matrix result = applyBi (other, (Matrix.MatrixBiOperation & Serializable) (value1, value2) -> value2 != 0 ? value1 / value2 : Double.MAX_VALUE);
        addToProcedureFactory(other, result, Expression.Type.DIV);
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
        addToProcedureFactory(constantMatrix, result, Expression.Type.DIV);
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
        addToProcedureFactory(constantMatrix, result, Expression.Type.DIV);
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
        return apply ((Matrix.MatrixUniOperation & Serializable) (value) -> Math.pow(value, power));
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
        apply (result, (Matrix.MatrixUniOperation & Serializable) (value) -> Math.pow(value, power));
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
        applyBi (other, result, (Matrix.MatrixBiOperation & Serializable) Math::max);
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
        return applyBi (other, (Matrix.MatrixBiOperation & Serializable) Math::max);
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
        applyBi (other, result, (Matrix.MatrixBiOperation & Serializable) Math::min);
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
        return applyBi (other, (Matrix.MatrixBiOperation & Serializable) Math::min);
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
        applyBi (other, result, (Matrix.MatrixBiOperation & Serializable) (value1, value2) -> Math.signum(value1) * Math.signum(value2));
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
        return applyBi (other, (Matrix.MatrixBiOperation & Serializable) (value1, value2) -> Math.signum(value1) * Math.signum(value2));
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
        addToProcedureFactory(other, result, Expression.Type.DOT);
        if (!masked || other.isMasked()) {
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
                if (!getRowMask(row)) {
                    for (int col = 0; col < other.getCols(); col++) {
                        if (!other.getColMask(col)) {
                            for (int x = 0; x < getCols(); x++) {
                                if (!getMask(row, x) && !other.getMask(x, col)) {
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
     * Return result array which has first element containing cumulative sum and second element number of elements.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param operation single variable operation defined as lambda operator.
     * @return array containing cumulative sum and element count as elements.
     * @throws MatrixException not thrown in any situation.
     */
    private double[] count(MatrixUniOperation operation) throws MatrixException {
        double[] result = new double[2];
        result[0] = result[1] = 0;
        if (!masked) {
            for (int row = 0; row < getRows(); row++) {
                for (int col = 0; col < getCols(); col++) {
                    result[0] += operation.execute(getValue(row, col));
                    result[1]++;
                }
            }
        }
        else {
            for (int row = 0; row < getRows(); row++) {
                if (!getRowMask(row)) {
                    for (int col = 0; col < getCols(); col++) {
                        if (!getMask(row, col) && !getColMask(col)) {
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
     * @throws MatrixException not thrown in any situation.
     */
    public double sum() throws MatrixException {
        MatrixUniOperation operation = (Matrix.MatrixUniOperation & Serializable) value -> value;
        double[] result = count(operation);
        return result[0];
    }

    /**
     * Takes mean of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return mean of elements of this matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public double mean() throws MatrixException {
        MatrixUniOperation operation = (Matrix.MatrixUniOperation & Serializable) value -> value;
        double[] result = count(operation);
        return result[1] > 0 ? result[0] / result[1] : 0;
    }

    /**
     * Takes variance of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return variance of elements of this matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public double var() throws MatrixException {
        double mean = sum();
        MatrixUniOperation operation = (Matrix.MatrixUniOperation & Serializable) value -> Math.pow(value - mean, 2);
        double[] result = count(operation);
        return result[1] > 0 ? result[0] / result[1] : 0;
    }

    /**
     * Takes variance of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return variance of elements of this matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public double var(double mean) throws MatrixException {
        MatrixUniOperation operation = (Matrix.MatrixUniOperation & Serializable) value -> Math.pow(value - mean, 2);
        double[] result = count(operation);
        return result[1] > 0 ? result[0] / result[1] : 0;
    }

    /**
     * Takes standard deviation of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return standard deviation of elements of this matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public double std() throws MatrixException {
        double mean = sum();
        MatrixUniOperation operation = (Matrix.MatrixUniOperation & Serializable) value -> Math.pow(value - mean, 2);
        double[] result = count(operation);
        return result[1] > 0 ? Math.sqrt(result[0] / (result[1] - 1)) : 0;
    }

    /**
     * Takes standard deviation of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return standard deviation of elements of this matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public double std(double mean) throws MatrixException {
        MatrixUniOperation operation = (Matrix.MatrixUniOperation & Serializable) value -> Math.pow(value - mean, 2);
        double[] result = count(operation);
        return result[1] > 0 ? Math.sqrt(result[0] / (result[1] - 1)) : 0;
    }

    /**
     * Takes cumulative p- norm (p is number equal or bigger than 1) of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param p p value for norm.
     * @return cumulative norm value of matrix.
     * @throws MatrixException not thrown in any situation.
     */
    public double norm(int p) throws MatrixException {
        MatrixUniOperation operation = (Matrix.MatrixUniOperation & Serializable) value -> Math.pow(Math.abs(value), p);
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
        return other.apply(other, (Matrix.MatrixUniOperation & Serializable) (value) -> (value - mean) / std);
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
        return other.apply(other, (Matrix.MatrixUniOperation & Serializable) (value) -> (value - min) / delta * (newMax - newMin) + newMin);
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
        double value = min ? Double.MAX_VALUE : Double.MIN_VALUE;
        if (!masked) {
            for (int row = 0; row < getRows(); row++) {
                for (int col = 0; col < getCols(); col++) {
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
        else {
            for (int row = 0; row < getRows(); row++) {
                if (!getRowMask(row)) {
                    for (int col = 0; col < getCols(); col++) {
                        if (!getMask(row, col) && !getColMask(col)) {
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
     * Defines size of slice with specific row and column size.
     *
     * @param sliceRowSize slice size in rows.
     * @param sliceColSize slice size in columns.
     * @throws MatrixException throws exception if requested slice does not fit inside matrix.
     */
    public void setSliceSize(int sliceRowSize, int sliceColSize) throws MatrixException {
        if (sliceAtRow < 0 || sliceAtCol < 0 || sliceAtRow + sliceRowSize > getRows() || sliceAtCol + sliceColSize > getCols()) throw new MatrixException("Slice starting at " + sliceAtRow + "x" + sliceAtCol + " of size " + sliceRowSize + "x" + sliceColSize + " does not fit within matrix.");
        this.sliceRowSize = sliceRowSize;
        this.sliceColSize = sliceColSize;
    }

    /**
     * Defines slice at specific row and column.
     *
     * @param sliceAtRow slice at row
     * @param sliceAtCol slice at column
     * @return returns this matrix.
     * @throws MatrixException throws exception if requested slice does not fit inside matrix.
     */
    public Matrix sliceAt(int sliceAtRow, int sliceAtCol) throws MatrixException {
        if (sliceAtRow < 0 || sliceAtCol < 0 || sliceAtRow + sliceRowSize > getRows() || sliceAtCol + sliceColSize > getCols()) throw new MatrixException("Slice starting at " + sliceAtRow + "x" + sliceAtCol + " of size " + sliceRowSize + "x" + sliceColSize + " does not fit within matrix.");
        this.sliceAtRow = sliceAtRow;
        this.sliceAtCol = sliceAtCol;
        return this;
    }

    /**
     * Defines slice at specific row and column give specific row and column size.
     *
     * @param sliceAtRow slice at row
     * @param sliceAtCol slice at column
     * @param sliceRowSize slice size in rows.
     * @param sliceColSize slice size in columns.
     * @return returns this matrix.
     * @throws MatrixException throws exception if requested slice does not fit inside matrix.
     */
    public Matrix sliceAt(int sliceAtRow, int sliceAtCol, int sliceRowSize, int sliceColSize) throws MatrixException {
        if (sliceAtRow < 0 || sliceAtCol < 0 || sliceAtRow + sliceRowSize > getRows() || sliceAtCol + sliceColSize > getCols()) throw new MatrixException("Slice starting at " + sliceAtRow + "x" + sliceAtCol + " of size " + sliceRowSize + "x" + sliceColSize + " does not fit within matrix.");
        this.sliceAtRow = sliceAtRow;
        this.sliceAtCol = sliceAtCol;
        this.sliceRowSize = sliceRowSize;
        this.sliceColSize = sliceColSize;
        return this;
    }

    /**
     * Get row where slice is started at.
     *
     * @return row where slice is started at.
     */
    public int getSliceAtRow() {
        return sliceAtRow;
    }

    /**
     * Get column where slice is started at.
     *
     * @return column where slice is started at.
     */
    public int getSliceAtCol() {
        return sliceAtCol;
    }

    /**
     * Returns size of slice in rows.
     *
     * @return size of slice in rows.
     */
    public int getSliceRowSize() {
        return sliceRowSize;
    }

    /**
     * Returns size of slice in columns.
     *
     * @return size of slice in columns.
     */
    public int getSliceColSize() {
        return sliceColSize;
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated value of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double convolve(Matrix filter) throws MatrixException {
        double result = 0;
        if (!masked) {
            for (int row = 0; row < getSliceRowSize(); row++) {
                for (int col = 0; col < getSliceColSize(); col++) {
                    result += getValue(getSliceAtRow() + row, getSliceAtCol() + col) * filter.getValue(filter.getSliceAtRow() + filter.getSliceRowSize() - 1 - row, filter.getSliceAtCol() + filter.getSliceColSize() - 1 - col);
                }
            }
        }
        else {
            for (int row = 0; row < getSliceRowSize(); row++) {
                if (!getRowMask(row) && !filter.getRowMask(row)) {
                    for (int col = 0; col < getSliceColSize(); col++) {
                        if (!getMask(row, col) && !getColMask(col) && !filter.getMask(row, col) && !filter.getColMask(col)) {
                            result += getValue(getSliceAtRow() + row, getSliceAtCol() + col) * filter.getValue(filter.getSliceAtRow() + filter.getSliceRowSize() - 1 - row, filter.getSliceAtCol() + filter.getSliceColSize() - 1 - col);
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * Calculates gradient of convolution.
     *
     * @param gradValue inner gradient value.
     * @param result gradient of convolution operation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void convolveGrad(double gradValue, Matrix result) throws MatrixException {
        if (!masked) {
            for (int row = 0; row < getSliceRowSize(); row++) {
                for (int col = 0; col < getSliceColSize(); col++) {
                    result.addValue(result.getSliceAtRow() + row, result.getSliceAtCol() + col, getValue(getSliceAtRow() + getSliceRowSize() - 1 - row, getSliceAtCol() + getSliceColSize() - 1 - col) * gradValue);
                }
            }
        }
        else {
            for (int row = 0; row < getSliceRowSize(); row++) {
                if (!getRowMask(row)) {
                    for (int col = 0; col < getSliceColSize(); col++) {
                        if (!getMask(row, col) && !getColMask(col)) {
                            result.addValue(result.getSliceAtRow() + row, result.getSliceAtCol() + col, getValue(getSliceAtRow() + getSliceRowSize() - 1 - row, getSliceAtCol() + getSliceColSize() - 1 - col) * gradValue);
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
    public double crosscorrelate(Matrix filter) throws MatrixException {
        double result = 0;
        if (!masked) {
            for (int row = 0; row < getSliceRowSize(); row++) {
                for (int col = 0; col < getSliceColSize(); col++) {
                    result += getValue(getSliceAtRow() + row, getSliceAtCol() + col) * filter.getValue(filter.getSliceAtRow() + row, filter.getSliceAtCol() + col);
                }
            }
        }
        else {
            for (int row = 0; row < getSliceRowSize(); row++) {
                if (!getRowMask(row) && !filter.getRowMask(row)) {
                    for (int col = 0; col < getSliceColSize(); col++) {
                        if (!getMask(row, col) && !getColMask(col) && !filter.getMask(row, col) && !filter.getColMask(col)) {
                            result += getValue(getSliceAtRow() + row, getSliceAtCol() + col) * filter.getValue(filter.getSliceAtRow() + row, filter.getSliceAtCol() + col);
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * Calculates gradient of cross-correlation.
     *
     * @param gradValue inner gradient value.
     * @param result gradient of cross-correlation operation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void crosscorrelateGrad(double gradValue, Matrix result) throws MatrixException {
        if (!masked) {
            for (int row = 0; row < getSliceRowSize(); row++) {
                for (int col = 0; col < getSliceColSize(); col++) {
                    result.addValue(result.getSliceAtRow() + row, result.getSliceAtCol() + col, getValue(getSliceAtRow() + row, getSliceAtCol() + col) * gradValue);
                }
            }
        }
        else {
            for (int row = 0; row < getSliceRowSize(); row++) {
                if (!getRowMask(row)) {
                    for (int col = 0; col < getSliceColSize(); col++) {
                        if (!getMask(row, col) && !getColMask(col)) {
                            result.addValue(result.getSliceAtRow() + row, result.getSliceAtCol() + col, getValue(getSliceAtRow() + row, getSliceAtCol() + col) * gradValue);
                        }
                    }
                }
            }
        }
    }

    /**
     * Executes max pooling operation for the matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param result result matrix of max pooling operation.
     * @param stride stride of convolution operation.
     * @param poolSize size of pool for max pooling operation.
     * @param maxargs calculated max pooling positions. Two first dimensions represent row and col of gradient for previous layer. Third argument stores row and col information of gradient for next layer.
     * @return result matrix of max pooling operation.
     * @throws MatrixException thrown if dimensions of matrices are not matching for calculation.
     */
    public Matrix maxPool(Matrix result, int stride, int poolSize, int[][][] maxargs) throws MatrixException {
        int sourceRows = getRows();
        int sourceCols = getCols();
        int resultRows = result.getRows();
        int resultCols = result.getCols();
        if (result.getRows() + poolSize - 1 != sourceRows || result.getCols() + poolSize - 1 != sourceCols) {
            throw new MatrixException("Dimensions of source (this) matrix: " + sourceRows + "x" + sourceCols + " poolSize: " + (poolSize - 1) + "x" + (poolSize - 1) + " and result matrix: " + result.getRows() + "x" + result.getCols() + " are not matching.");
        }
        if (!masked) {
            for (int resultRow = 0; resultRow < resultRows; resultRow++) {
                for (int resultCol = 0; resultCol < resultCols; resultCol++) {
                    double maxValue = Double.MIN_VALUE;
                    for (int targetRow = 0; targetRow < poolSize - 1; targetRow++) {
                        for (int targetCol = 0; targetCol < poolSize - 1; targetCol++) {
                            int sourceRow = stride * resultRow + targetRow;
                            int sourceCol = stride * resultCol + targetCol;
                            double curValue = getValue(sourceRow, sourceCol);
                            if (maxValue < curValue) {
                                maxValue = curValue;
                                maxargs[resultRow][resultCol][0] = sourceRow;
                                maxargs[resultRow][resultCol][1] = sourceCol;
                            }
                        }
                    }
                    result.setValue(resultRow, resultCol, maxValue);
                }
            }
        }
        else {
            for (int resultRow = 0; resultRow < result.getRows(); resultRow++) {
                for (int resultCol = 0; resultCol < result.getCols(); resultCol++) {
                    double maxValue = Double.MIN_VALUE;
                    maxargs[resultRow][resultCol][0] = -1;
                    maxargs[resultRow][resultCol][1] = -1;
                    for (int targetRow = 0; targetRow < poolSize - 1; targetRow++) {
                        for (int targetCol = 0; targetCol < poolSize - 1; targetCol++) {
                            int sourceRow = stride * resultRow + targetRow;
                            int sourceCol = stride * resultCol + targetCol;
                            double curValue = getValue(sourceRow, sourceCol);
                            boolean maskedEntry = isMasked(sourceRow, sourceCol);
                            if (!maskedEntry && maxValue < curValue) {
                                maxValue = curValue;
                                maxargs[resultRow][resultCol][0] = sourceRow;
                                maxargs[resultRow][resultCol][1] = sourceCol;
                            }
                        }
                    }
                    if (maxValue != Double.MIN_VALUE) result.setValue(resultRow, resultCol, maxValue);
                }
            }
        }
        return result;
    }

    /**
     * Calculates gradients for max pool operation. Assigns backward gradient value from position (int[0] as row, int[1] as col) calculated by max pool operation.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param dEoP result matrix of max pooling operation. This represent gradient towards previous layer.
     * @param maxargs calculated max pooling positions. Two first dimensions represent row and col of gradient for previous layer. Third argument stores row and col information of gradient for next layer.
     * @return result matrix of max pooling operation.
     */
    public Matrix maxPoolGrad(Matrix dEoP, int[][][] maxargs) {
        if (!masked) {
            for (int row = 0; row < maxargs.length; row++) {
                for (int col = 0; col < maxargs.length; col++) {
                    dEoP.setValue(maxargs[row][col][0], maxargs[row][col][1], getValue(row, col));
                }
            }
        }
        else {
            for (int row = 0; row < maxargs.length; row++) {
                for (int col = 0; col < maxargs.length; col++) {
                    if (maxargs[row][col][0] != -1) dEoP.setValue(maxargs[row][col][0], maxargs[row][col][1], getValue(row, col));
                }
            }
        }
        return dEoP;
    }

    /**
     * Executes average pooling operation for the matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param result result matrix of average pooling operation.
     * @param stride stride of convolution operation.
     * @param poolSize size of pool for average pooling operation.
     * @return result matrix of average pooling operation.
     * @throws MatrixException thrown if dimensions of matrices are not matching for calculation.
     */
    public Matrix avgPool(Matrix result, int stride, int poolSize) throws MatrixException {
        int sourceRows = getRows();
        int sourceCols = getCols();
        if (result.getRows() + poolSize - 1 != sourceRows || result.getCols() + poolSize - 1 != sourceCols) {
            throw new MatrixException("Dimensions of source (this) matrix: " + sourceRows + "x" + sourceCols + " poolSize: " + (poolSize - 1) + "x" + (poolSize - 1) + " and result matrix: " + result.getRows() + "x" + result.getCols() + " are not matching.");
        }
        double invAmount = 1 / (double)(poolSize * poolSize);
        if (!masked) {
            for (int resultRow = 0; resultRow < result.getRows(); resultRow++) {
                for (int resultCol = 0; resultCol < result.getCols(); resultCol++) {
                    double value = 0;
                    for (int targetRow = 0; targetRow < poolSize - 1; targetRow++) {
                        for (int targetCol = 0; targetCol < poolSize - 1; targetCol++) {
                            int sourceRow = stride * resultRow + targetRow;
                            int sourceCol = stride * resultCol + targetCol;
                            value += getValue(sourceRow, sourceCol);
                        }
                    }
                    result.setValue(resultRow, resultCol, value * invAmount);
                }
            }
        }
        else {
            for (int resultRow = 0; resultRow < result.getRows(); resultRow++) {
                for (int resultCol = 0; resultCol < result.getCols(); resultCol++) {
                    double value = 0;
                    for (int targetRow = 0; targetRow < poolSize - 1; targetRow++) {
                        for (int targetCol = 0; targetCol < poolSize - 1; targetCol++) {
                            int sourceRow = stride * resultRow + targetRow;
                            int sourceCol = stride * resultCol + targetCol;
                            boolean maskedEntry = isMasked(sourceRow, sourceCol);
                            if (!maskedEntry) value += getValue(sourceRow, sourceCol);
                        }
                    }
                    result.setValue(resultRow, resultCol, value * invAmount);
                }
            }
        }
        return result;
    }

    /**
     * Calculates gradients for average pool operation. Assigns backward gradient value of 1 / pool size^2 for each position of gradient towards previous layer.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param dEoP result matrix of average pooling operation. This represent gradient towards previous layer.
     * @param poolSize size of pool for average pooling operation.
     * @return result matrix of average pooling operation.
     */
    public Matrix avgPoolGrad(Matrix dEoP, int poolSize) {
        double value = 1 / (double)(poolSize * poolSize);
        for (int row = 0; row < dEoP.getRows(); row++) {
            for (int col = 0; col < dEoP.getRows(); col++) {
                dEoP.setValue(row, col, value);
            }
        }
        return dEoP;
    }

    /**
     * Transposes matrix.
     *
     * @return reference to this matrix but with transposed that is flipped rows and columns.
     */
    public Matrix T() {
        try {
            // Make shallow copy of matrix leaving references internal objects which are shared.
            Matrix clone = (Matrix)clone();
            clone.t = !clone.t; // transpose
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
     */
    public void setMask() {
        masked = true;
        resetMask();
    }

    /**
     * Removes mask from this matrix.
     *
     */
    public void unsetMask() {
        masked = false;
        noMask();
    }

    /**
     * Clears and removes mask from this matrix.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     */
    protected abstract void noMask();

    /**
     * Checks if matrix is masked.
     *
     * @return true is matrix is masked otherwise false.
     */
    public boolean isMasked() {
        return masked;
    }

    /**
     * Checks if matrix is masked at specific row and / or col
     *
     * @param row row to be checked.
     * @param col col to be checked.
     * @return result of mask check.
     * @throws MatrixException throws exception if row and / or col are beyond matrix boundaries.
     */
    public boolean isMasked(int row, int col) throws MatrixException {
        return getRowMask(row) || getColMask(col) || getMask(row, col);
    }

    /**
     * Checks if mask is set.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @throws MatrixException throws exception if mask is not set.
     */
    protected abstract void checkMask() throws MatrixException;

    /**
     * Resets current mask.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     */
    protected abstract void resetMask();

    /**
     * Sets bernoulli probability to mask specific element of this matrix.
     *
     * @param proba masking probability between 0 (0%) and 1 (100%).
     * @throws MatrixException throws exception if masking probability is not between 0 and 1.
     */
    public void setMaskProba(double proba) throws MatrixException {
        if (proba < 0 || proba > 1) throw new MatrixException("Masking probability must be between 0 and 1.");
        this.proba = proba;
    }

    /**
     * Returns current bernoulli masking probability.
     *
     * @return masking probability.
     */
    public double getMaskProba() {
        return proba;
    }

    /**
     * Pushes current mask into stack and optionally creates new mask for this matrix.<br>
     * Useful in operations where sequence of operations and taken between this matrix and other matrices.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     * @throws MatrixException throws exception if mask is not set.
     */
    public abstract void stackMask(boolean reset) throws MatrixException;

    /**
     * Pops mask from mask stack.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @throws MatrixException throws exception if mask stack is empty.
     */
    public abstract void unstackMask() throws MatrixException;

    /**
     * Returns size of a mask stack.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @return size of mask stack.
     */
    public abstract int maskStackSize();

    /**
     * Clears mask stack.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     */
    public abstract void clearMaskStack();

    // Return false by probability proba

    /**
     * Returns true with probability of proba. Proba is masking probability of this matrix.
     *
     * @return true with probability of proba.
     */
    private boolean maskByProbability() {
        return random.nextDouble() > proba;
    }

    /**
     * Matrix internal function used to set matrix masking of specific row and column.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @param row row of value to be set.
     * @param col column of value to be set.
     * @param value defines if specific matrix row and column is masked (true) or not (false).
     */
    protected abstract void setMaskValue(int row, int col, boolean value);

    /**
     * Matrix internal function used to get matrix masking of specific row and column.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @param row row of value to be returned.
     * @param col column of value to be returned.
     * @return if specific matrix row and column if masked (true) or not (false).
     */
    protected abstract boolean getMaskValue(int row, int col);

    /**
     * Sets mask for element at specific row and column.
     *
     * @param row row of mask to be set.
     * @param col column of mask to be set.
     * @param value sets mask if true otherwise unsets mask.
     * @throws MatrixException throws exception is given row or column is beyond dimensions of this matrix.
     */
    public void setMask(int row, int col, boolean value) throws MatrixException {
        checkMask();
        setMaskValue(row, col, value);
    }

    /**
     * Gets mask for element at specific row and column.
     *
     * @param row row of mask to be returned.
     * @param col column of mask to be returned.
     * @return true if mask is set otherwise false.
     * @throws MatrixException throws exception is given row or column is beyond dimensions of this matrix.
     */
    public boolean getMask(int row, int col) throws MatrixException {
        if (!isMasked()) return false;
        checkMask();
        return getMaskValue(row, col);
    }

    /**
     * Sets masking for this matrix with given bernoulli probability proba.
     *
     * @throws MatrixException not thrown in any situation.
     */
    public void maskByProba() throws MatrixException {
        checkMask();
        for (int row = 0; row < getRows(); row++) {
            for (int col = 0; col < getCols(); col++) {
                // Mask out (set value as true) true by probability of proba
                if (maskByProbability()) setMask(row, col, true);
            }
        }
    }

    /**
     * Checks if row mask is set.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @throws MatrixException throws exception if row mask is not set.
     */
    protected abstract void checkRowMask() throws MatrixException;

    /**
     * Pushes current row mask into stack and optionally creates new mask for this matrix.<br>
     * Useful in operations where sequence of operations and taken between this matrix and other matrices.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     * @throws MatrixException throws exception if mask is not set.
     */
    public abstract void stackRowMask(boolean reset) throws MatrixException;

    /**
     * Pops row mask from mask stack.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @throws MatrixException throws exception if row mask stack is empty.
     */
    public abstract void unstackRowMask() throws MatrixException;

    /**
     * Returns size of a row mask stack.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @return size of row mask stack.
     */
    public abstract int rowMaskStackSize();

    /**
     * Clears row mask stack.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     */
    public abstract void clearRowMaskStack();

    /**
     * Sets mask value for row mask.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @param row row of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     * @throws MatrixException throws exception if row is beyond dimension of current row mask.
     */
    protected abstract void setRowMaskValue(int row, boolean value) throws MatrixException;

    /**
     * Sets mask value for row mask.
     *
     * @param row row of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     * @throws MatrixException throws exception if row is beyond dimension of current row mask.
     */
    public void setRowMask(int row, boolean value) throws MatrixException {
        checkMask();
        setRowMaskValue(row, value);
    }

    /**
     * Gets mask value for row mask.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @param row row of mask to be returned.
     * @return true if row mask is set otherwise false.
     * @throws MatrixException throws exception if row is beyond dimension of current row mask or row mask is not set.
     */
    protected abstract boolean getRowMaskValue(int row) throws MatrixException;

    /**
     * Gets mask value for row mask.
     *
     * @param row row of mask to be returned.
     * @return true if row mask is set otherwise false.
     * @throws MatrixException throws exception if row is beyond dimension of current row mask or row mask is not set.
     */
    public boolean getRowMask(int row) throws MatrixException {
        return getRowMaskValue(row);
    }

    /**
     * Sets row masking for this matrix with given bernoulli probability proba.
     *
     * @throws MatrixException not thrown in any situation.
     */
    public void maskRowByProba() throws MatrixException {
        checkMask();
        for (int row = 0; row < getRows(); row++) {
            // Mask out (set value as true) by probability of proba
            if (maskByProbability()) setRowMask(row, true);
        }
    }

    /**
     * Checks if column mask is set.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @throws MatrixException throws exception if column mask is not set.
     */
    protected abstract void checkColMask() throws MatrixException;

    /**
     * Sets mask value for column mask.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @param col column of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     * @throws MatrixException throws exception if column is beyond dimension of current column mask or column mask is not set.
     */
    protected abstract void setColMaskValue(int col, boolean value) throws MatrixException;

    /**
     * Sets mask value for column mask.
     *
     * @param col column of mask to be set.
     * @param value if true sets row mask otherwise unsets mask.
     * @throws MatrixException throws exception if column is beyond dimension of current column mask or column mask is not set.
     */
    public void setColMask(int col, boolean value) throws MatrixException {
        setColMaskValue(col, value);
    }

    /**
     * Sets column masking for this matrix with given bernoulli probability proba.
     *
     * @throws MatrixException not thrown in any situation.
     */
    public void maskColByProba() throws MatrixException {
        checkMask();
        for (int col = 0; col < getCols(); col++) {
            // Mask out (set value as true) true by probability of proba
            if (maskByProbability()) setColMask(col, true);
        }
    }

    /**
     * Gets mask value for column mask.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @param col column of mask to be returned.
     * @return true if row mask is set otherwise false.
     * @throws MatrixException throws exception if column is beyond dimension of current column mask.
     */
    protected abstract boolean getColMaskValue(int col) throws MatrixException;

    /**
     * Gets mask value for column mask.
     *
     * @param col column of mask to be returned.
     * @return true if row mask is set otherwise false.
     * @throws MatrixException throws exception if column is beyond dimension of current column mask or mask is not set.
     */
    public boolean getColMask(int col) throws MatrixException {
        if (!isMasked()) return false;
        checkMask();
        return getColMaskValue(col);
    }

    /**
     * Pushes current column mask into stack and optionally creates new mask for this matrix.<br>
     * Useful in operations where sequence of operations and taken between this matrix and other matrices.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @param reset if true new mask is generated after current mask is stacked.
     * @throws MatrixException throws exception if mask is not set.
     */
    public abstract void stackColMask(boolean reset) throws MatrixException;

    /**
     * Pops column mask from mask stack.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @throws MatrixException throws exception if column mask stack is empty.
     */
    public abstract void unstackColMask() throws MatrixException;

    /**
     * Returns size of a column mask stack.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     * @return size of column mask stack.
     */
    public abstract int colMaskStackSize();

    /**
     * Clears column mask stack.<br>
     * Abstract function to be implemented by underlying matrix data structure class implementation.<br>
     * This is typically dense matrix (DMatrix class) or sparse matrix (SMatrix class).<br>
     *
     */
    public abstract void clearColMaskStack();

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
