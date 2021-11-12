/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.matrix;

import utils.configurable.DynamicParamException;
import utils.procedure.ProcedureFactory;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Interface that implements matrix with extensive set of matrix operations and masking for matrix.<br>
 *
 */
public interface Matrix {

    /**
     * Sets name for matrix.
     *
     * @param name matrix name.
     */
    void setName(String name);

    /**
     * Returns name of matrix.
     *
     * @return name of matrix.
     */
    String getName();

    /**
     * Defines interface to be used as part of lambda function to initialize Matrix.
     */
    interface Initializer {

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
     * Defines interface to be used as part of lambda function to execute single argument matrix operation.
     */
    interface MatrixUnaryOperation {

        /**
         * Defines operation to be executed with single parameter.
         *
         * @param value1 value for parameter.
         * @return value returned by the operation.
         */
        double execute(double value1);
    }

    /**
     * Defines interface to be used as part of lambda function to execute two argument matrix operation.
     */
    interface MatrixBinaryOperation {

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
     * Resets matrix and it's mask.
     *
     */
    void reset();

    /**
     * Returns sub-matrices within Matrix.
     *
     * @return sub-matrices within Matrix.
     */
    ArrayList<Matrix> getSubMatrices();

    /**
     * Sets initializer of matrix.
     *
     * @param initializer initializer.
     */
    void setInitializer(Initializer initializer);

    /**
     * Returns initializer of matrix.
     *
     * @return initializer.
     */
    Matrix.Initializer getInitializer();

    /**
     * Initializes matrix.
     *
     * @param initialization type of initialization defined in class Init.
     */
    void initialize(Initialization initialization);

    /**
     * Initializes matrix.
     *
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     */
    void initialize(Initialization initialization, int inputs, int outputs);

    /**
     * Returns true if matrix is scalar otherwise false.
     *
     * @return true if matrix is scalar otherwise false.
     */
    boolean isScalar();

    /**
     * Initializes matrix with given initializer operation.
     *
     * @param initializer initializer operation.
     */
    void initialize(Initializer initializer);

    /**
     * Initializes matrix with given value.
     *
     * @param value initialization value.
     */
    void initializeToValue(double value);

    /**
     * Sets value of matrix at specific row and column.
     *
     * @param row row of value to be set.
     * @param column column of value to be set.
     * @param value new value to be set.
     */
    void setValue(int row, int column, double value);

    /**
     * Returns value of matrix at specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @return value of row and column.
     */
    double getValue(int row, int column);

    /**
     * Returns total number of rows defined for matrix.
     *
     * @return total number of rows defined for matrix.
     */
    int getTotalRows();

    /**
     * Returns total number of columns defined for matrix.
     *
     * @return total number of columns defined for matrix.
     */
    int getTotalColumns();

    /**
     * Returns size (rows * columns) of matrix effectively with slicing considered.<br>
     *
     * @return size (rows * columns) of matrix effectively with slicing considered.
     */
    int size();

    /**
     * Returns number of rows in matrix effectively with slicing considered.<br>
     *
     * @return number of rows in matrix effectively with slicing considered.
     */
    int getRows();

    /**
     * Returns number of columns in matrix effectively with slicing considered.<br>
     *
     * @return number of columns in matrix effectively with slicing considered.
     */
    int getColumns();

    /**
     * Increment value of specific row and column.
     *
     * @param row row of value to be added.
     * @param column column of value to be added.
     * @param value to be added.
     */
    void incrementByValue(int row, int column, double value);

    /**
     * Decrease value of specific row and column.
     *
     * @param row row of value to be decreased.
     * @param column column of value to be decreased.
     * @param value to be decreased.
     */
    void decrementByValue(int row, int column, double value);

    /**
     * Multiply value of specific row and column.
     *
     * @param row row of value to be multiplied.
     * @param column column of value to be multiplied.
     * @param value to be multiplied.
     */
    void multiplyByValue(int row, int column, double value);

    /**
     * Divide value of specific row and column.
     *
     * @param row row of value to be divided.
     * @param column column of value to be divided.
     * @param value to be divided.
     */
    void divideByValue(int row, int column, double value);

    /**
     * Returns new matrix of same dimensions.
     *
     * @return new matrix of same dimensions.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    Matrix getNewMatrix() throws MatrixException;

    /**
     * Returns new matrix of same dimensions optionally as transposed.
     *
     * @param asTransposed if true returns new matrix as transposed otherwise with unchanged dimensions.
     * @return new matrix of same dimensions.
     */
    Matrix getNewMatrix(boolean asTransposed);

    /**
     * Copies new matrix data inside this matrix same dimensions.<br>
     *
     * @param newMatrix new matrix to be copied inside this matrix.
     * @throws MatrixException throws exception if this and new matrix dimensions are not matching.
     */
    void copyMatrixData(Matrix newMatrix) throws MatrixException;

    /**
     * Creates new matrix with object reference to the matrix data of this matrix.
     *
     * @return newly created reference matrix.
     * @throws MatrixException throws exception if mask operation fails or cloning of matrix fails.
     */
    Matrix reference() throws MatrixException;

    /**
     * Creates new matrix with object full copy of this matrix.
     *
     * @return newly created reference matrix.
     * @throws MatrixException throws exception if mask is not set or cloning of matrix fails.
     */
    Matrix copy() throws MatrixException;

    /**
     * Slices current matrix by creating reference to existing matrix.
     *
     * @param startRow start row of slice.
     * @param startColumn start column of slice.
     * @param endRow  end row of slice.
     * @param endColumn  end column of slice.
     * @return sliced matrix.
     * @throws MatrixException throws exception if slicing fails.
     */
    Matrix slice(int startRow, int startColumn, int endRow, int endColumn) throws MatrixException;

    /**
     * Slices matrix.
     *
     * @param startRow start row of slice.
     * @param startColumn start column of slice.
     * @param endRow  end row of slice.
     * @param endColumn  end column of slice.
     * @throws MatrixException throws exception if slicing fails.
     */
    void sliceAt(int startRow, int startColumn, int endRow, int endColumn) throws MatrixException;

    /**
     * Removes slicing of matrix.
     *
     */
    void unslice();

    /**
     * Checks if this matrix and other matrix are equal in dimensions (rows x columns).
     *
     * @param other other matrix to be compared against.
     * @return true if matrices are of same size otherwise false.
     */
    boolean hasEqualSize(Matrix other);

    /**
     * Sets procedure factory for matrix.
     *
     * @param procedureFactory new procedure factory.
     */
    void setProcedureFactory(ProcedureFactory procedureFactory);

    /**
     * Returns current procedure factory of matrix.
     *
     * @return current procedure factory.
     */
    ProcedureFactory getProcedureFactory();

    /**
     * Removes procedure factory.
     *
     */
    void removeProcedureFactory();

    /**
     * Returns true if matrix has procedure factory otherwise false.
     *
     * @return true if matrix has procedure factory otherwise false.
     */
    boolean hasProcedureFactory();

    /**
     * Sets flag if matrix is normalized.
     *
     * @param normalize if true matrix is normalized.
     */
    void setNormalize(boolean normalize);

    /**
     * Returns flag if matrix is normalized.
     *
     * @return if true matrix is normalized.
     */
    boolean isNormalized();

    /**
     * Sets flag if matrix is regularized.
     *
     * @param regularize if true matrix is regularized.
     */
    void setRegularize(boolean regularize);

    /**
     * Returns flag if matrix is regularized.
     *
     * @return if true matrix is regularized.
     */
    boolean isRegularized();

    /**
     * Makes current matrix data equal to other matrix data.
     *
     * @param other other matrix to be copied as data of this matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    void setEqualTo(Matrix other) throws MatrixException;

    /**
     * Checks if data of other matrix is equal to data of this matrix
     *
     * @param other matrix to be compared.
     * @return true is data of this and other matrix are equal otherwise false.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    boolean equals(Matrix other) throws MatrixException;

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param result matrix which stores operation result.
     * @param matrixUnaryOperation single variable operation defined as lambda operator.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and result matrix are not of equal dimensions.
     */
    Matrix apply(Matrix result, MatrixUnaryOperation matrixUnaryOperation) throws MatrixException;

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param matrixUnaryOperation single variable operation defined as lambda operator.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    Matrix apply(MatrixUnaryOperation matrixUnaryOperation) throws MatrixException;

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param matrixUnaryOperation single variable operation defined as lambda operator.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    Matrix apply(MatrixUnaryOperation matrixUnaryOperation, boolean inplace) throws MatrixException;

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param result result matrix.
     * @param unaryFunction unaryFunction to be applied.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void apply(Matrix result, UnaryFunction unaryFunction) throws MatrixException;

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param unaryFunction unaryFunction to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    Matrix apply(UnaryFunction unaryFunction) throws MatrixException;

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param result result matrix
     * @param unaryFunctionType unaryFunction type to be applied.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void apply(Matrix result, UnaryFunctionType unaryFunctionType) throws MatrixException, DynamicParamException;

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
    Matrix apply(UnaryFunctionType unaryFunctionType) throws MatrixException, DynamicParamException;

    /**
     * Applies two variable operation to this matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @param matrixBinaryOperation two variable operation defined as lambda operator.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     */
    Matrix applyBi(Matrix other, Matrix result, Matrix.MatrixBinaryOperation matrixBinaryOperation) throws MatrixException;

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
    Matrix applyBi(Matrix other, Matrix.MatrixBinaryOperation matrixBinaryOperation) throws MatrixException;

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
    void applyBi(Matrix other, Matrix result, BinaryFunction binaryFunction) throws MatrixException;

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
    Matrix applyBi(Matrix other, BinaryFunction binaryFunction) throws MatrixException;

    /**
     * Applies two variable operation to this matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other other matrix
     * @param result result matrix.
     * @param binaryFunctionType binaryFunction type to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix applyBi(Matrix other, Matrix result, BinaryFunctionType binaryFunctionType) throws MatrixException, DynamicParamException;

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
    Matrix applyBi(Matrix other, BinaryFunctionType binaryFunctionType) throws MatrixException, DynamicParamException;

    /**
     * Adds other matrix to this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    void add(Matrix other, Matrix result) throws MatrixException;

    /**
     * Adds other matrix to this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    Matrix add(Matrix other) throws MatrixException;

    /**
     * Adds constant number to this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant contains constant value to be added.
     * @param result matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    void add(double constant, Matrix result) throws MatrixException;

    /**
     * Adds constant number to this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant contains constant value to be added.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    Matrix add(double constant) throws MatrixException;

    /**
     * Subtracts other matrix from this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    void subtract(Matrix other, Matrix result) throws MatrixException;

    /**
     * Subtracts other matrix from this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    Matrix subtract(Matrix other) throws MatrixException;

    /**
     * Subtracts constant number from this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant contains constant value to be subtracted.
     * @param result matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    void subtract(double constant, Matrix result) throws MatrixException;

    /**
     * Subtracts constant number from this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant contains constant value to be subtracted.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    Matrix subtract(double constant) throws MatrixException;

    /**
     * Multiplies other matrix element wise with this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    void multiply(Matrix other, Matrix result) throws MatrixException;

    /**
     * Multiplies other matrix element wise with this matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    Matrix multiply(Matrix other) throws MatrixException;

    /**
     * Multiplies constant number with this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant contains constant value to be multiplied.
     * @param result matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    void multiply(double constant, Matrix result) throws MatrixException;

    /**
     * Multiplies constant number with this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant contains constant value to be multiplied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    Matrix multiply(double constant) throws MatrixException;

    /**
     * Divides this matrix element wise with other matrix.<br>
     * In case any element value of other matrix is zero result is treated as Double MAX value to avoid NaN condition.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     */
    void divide(Matrix other, Matrix result) throws MatrixException;

    /**
     * Divides this matrix element wise with other matrix.<br>
     * In case any element value of other matrix is zero result is treated as Double MAX value to avoid NaN condition.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    Matrix divide(Matrix other) throws MatrixException;

    /**
     * Divides this matrix element wise with constant.<br>
     * In case constant is zero result is treated as Double MAX value to avoid NaN condition.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant constant used as divider value.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and result matrix are not of equal dimensions.
     */
    void divide(double constant, Matrix result) throws MatrixException;

    /**
     * Divides this matrix element wise with constant.<br>
     * In case constant is zero result is treated as Double MAX value to avoid NaN condition.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant constant used as divider value.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    Matrix divide(double constant) throws MatrixException;

    /**
     * Raises this matrix element wise to the power of value power.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param power power value to which this elements is to be raised.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix power(double power) throws MatrixException, DynamicParamException;

    /**
     * Raises this matrix element wise to the power of value power.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param power power value to which this elements is to be raised.
     * @param result matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void power(double power, Matrix result) throws MatrixException, DynamicParamException;

    /**
     * Takes element wise max value of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void max(Matrix other, Matrix result) throws MatrixException, DynamicParamException;

    /**
     * Takes element wise max value of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix max(Matrix other) throws MatrixException, DynamicParamException;

    /**
     * Takes element wise min value of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void min(Matrix other, Matrix result) throws MatrixException, DynamicParamException;

    /**
     * Takes element wise min value of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix min(Matrix other) throws MatrixException, DynamicParamException;

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
    void sgnmul(Matrix other, Matrix result) throws MatrixException, DynamicParamException;

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
    Matrix sgnmul(Matrix other) throws MatrixException, DynamicParamException;

    /**
     * Takes matrix dot product of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if columns of this matrix and rows of other matrix are not matching or rows of this and result matrix or columns of result and other matrix are not matching.
     */
    Matrix dot(Matrix other, Matrix result) throws MatrixException;

    /**
     *
     * Takes matrix dot product of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if columns of this matrix and rows of other matrix are not matching are not matching.
     */
    Matrix dot(Matrix other) throws MatrixException;

    /**
     * Returns constant as matrix.
     *
     * @param constant constant value.
     * @return constant matrix.
     */
    Matrix constantAsMatrix(double constant);

    /**
     * Takes element wise cumulative sum of this matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @return sum of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    double sum() throws MatrixException;

    /**
     * Takes element wise cumulative sum of this matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @return sum of matrix.
     * @throws MatrixException not thrown in any situation.
     */
    Matrix sumAsMatrix() throws MatrixException;

    /**
     * Takes mean of elements of this matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @return mean of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    double mean() throws MatrixException;

    /**
     * Takes mean of elements of this matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException not thrown in any situation.
     * @return mean of matrix.
     */
    Matrix meanAsMatrix() throws MatrixException;

    /**
     * Takes variance of elements of this matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @return variance of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    double variance() throws MatrixException;

    /**
     * Takes variance of elements of this matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException not thrown in any situation.
     * @return variance of matrix.
     */
    Matrix varianceAsMatrix() throws MatrixException;

    /**
     * Takes variance of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return variance of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    double variance(double mean) throws MatrixException;

    /**
     * Takes variance of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return variance of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix varianceAsMatrix(Matrix mean) throws MatrixException;

    /**
     * Takes standard deviation of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return standard deviation of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    double standardDeviation() throws MatrixException;

    /**
     * Takes standard deviation of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @throws MatrixException not thrown in any situation.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @return standard deviation of matrix.
     */
    Matrix standardDeviationAsMatrix() throws MatrixException, DynamicParamException;

    /**
     * Takes standard deviation of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return standard deviation of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    double standardDeviation(double mean) throws MatrixException;

    /**
     * Takes standard deviation of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return standard deviation of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix standardDeviationAsMatrix(Matrix mean) throws MatrixException;

    /**
     * Takes cumulative p- norm (p is number equal or bigger than 1) of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param p p value for norm.
     * @return norm of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    double norm(int p) throws MatrixException;

    /**
     * Takes cumulative p- norm (p is number equal or bigger than 1) of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param p p value for norm.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return norm of matrix.
     */
    Matrix normAsMatrix(int p) throws MatrixException;

    /**
     * Calculates exponential moving average.
     *
     * @param currentAverage current average value
     * @param beta degree of weighting decrease for exponential moving average.
     * @return updated average with new average value included.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix exponentialMovingAverage(Matrix currentAverage, double beta) throws MatrixException;

    /**
     * Normalizes matrix by removing mean and variance.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @param inplace if true matrix is normalized in place otherwise copy of normalized matrix is returned.
     * @return normalized matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
     Matrix normalize(boolean inplace) throws MatrixException;

    /**
     * Normalizes (scales) this matrix to new min and max values.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param newMinimum new minimum value.
     * @param newMaximum new maximum value.
     * @throws MatrixException not thrown in any situation.
     */
    void minMax(double newMinimum, double newMaximum) throws MatrixException;

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
    Matrix minMax(Matrix other, double newMinimum, double newMaximum) throws MatrixException;

    /**
     * Returns minimum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return minimum value of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    double min() throws MatrixException;

    /**
     * Returns minimum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return minimum value of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix minAsMatrix() throws MatrixException;

    /**
     * Returns argmin meaning row and column of matrix containing minimum value.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return array containing row and column in this order that points to minimum value of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    int[] argmin() throws MatrixException;

    /**
     * Returns maximum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return maximum value of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    double max() throws MatrixException;

    /**
     * Returns maximum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return maximum value of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix maxAsMatrix() throws MatrixException;

    /**
     * Returns argmax meaning row and column of matrix containing maximum value.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return array containing row and column in this order that points to maximum value of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    int[] argmax() throws MatrixException;

    /**
     * Returns entropy of matrix.
     *
     * @return entropy of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    double entropy() throws MatrixException;

    /**
     * Returns entropy of matrix.
     *
     * @param asDistribution if true matrix is forced into distribution prior calculating entropy.
     * @return entropy of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    double entropy(boolean asDistribution) throws MatrixException;

    /**
     * Returns softmax of this matrix.
     *
     * @param result result matrix.
     * @return softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    Matrix softmax(Matrix result) throws MatrixException;

    /**
     * Returns softmax of this matrix.
     *
     * @return softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    Matrix softmax() throws MatrixException;

    /**
     * Returns Gumbel softmax of this matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @param result result matrix.
     * @return softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    Matrix gumbelSoftmax(Matrix result) throws MatrixException;

    /**
     * Returns Gumbel softmax of this matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @param result result matrix.
     * @param gumbelSoftmaxTau tau value for Gumbel Softmax.
     * @return Gumbel softmax matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    Matrix gumbelSoftmax(Matrix result, double gumbelSoftmaxTau) throws MatrixException;

    /**
     * Returns Gumbel softmax of this matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @return Gumbel softmax matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    Matrix gumbelSoftmax() throws MatrixException;

    /**
     * Returns Gumbel softmax of this matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @param gumbelSoftmaxTau tau value for Gumbel Softmax.
     * @return Gumbel softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    Matrix gumbelSoftmax(double gumbelSoftmaxTau) throws MatrixException;

    /**
     * Returns softmax gradient of this matrix.<br>
     * Assumes that input matrix is softmax result.<br>
     *
     * @param result result matrix.
     * @return softmax gradient of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    Matrix softmaxGrad(Matrix result) throws MatrixException;

    /**
     * Returns softmax gradient of this matrix.
     *
     * @return softmax gradient of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    Matrix softmaxGrad() throws MatrixException;

    /**
     * Sets stride size for convolution and pooling operations.
     *
     * @param stride stride size.
     */
    void setStride(int stride);

    /**
     * Returns stride size for convolution and pooling operations.
     *
     * @return stride size.
     */
    int getStride();

    /**
     * Sets dilation step size for convolution operations.
     *
     * @param dilation dilation step size.
     */
    void setDilation(int dilation);

    /**
     * Returns dilation step size for convolution operations.
     *
     * @return dilation step size.
     */
    int getDilation();

    /**
     * Sets filter row size for convolution and pooling operations.
     *
     * @param filterRowSize filter row size.
     */
    void setFilterRowSize(int filterRowSize);

    /**
     * Sets filter column size for convolution and pooling operations.
     *
     * @param filterColumnSize filter column size.
     */
    void setFilterColumnSize(int filterColumnSize);

    /**
     * Returns filter row size for convolution and pooling operations.
     *
     * @return filter row size for convolution and pooling operations.
     */
    int getFilterRowSize();

    /**
     * Returns filter column size for convolution and pooling operations.
     *
     * @return filter column size for convolution and pooling operations.
     */
    int getFilterColumnSize();

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated value of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix convolve(Matrix filter) throws MatrixException;

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated value of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void convolve(Matrix filter, Matrix result) throws MatrixException;

    /**
     * Calculates crosscorrelation between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated value of crosscorrelation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix crosscorrelate(Matrix filter) throws MatrixException;

    /**
     * Calculates crosscorrelate between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated value of crosscorrelate.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void crosscorrelate(Matrix filter, Matrix result) throws MatrixException;

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated value of Winograd convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix winogradConvolve(Matrix filter) throws MatrixException;

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated value of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void winogradConvolve(Matrix filter, Matrix result) throws MatrixException;

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
    Matrix winogradConvolve(Matrix filter, Matrix A, Matrix AT, Matrix C, Matrix CT, Matrix G, Matrix GT) throws MatrixException;

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
    void winogradConvolve(Matrix filter, Matrix result, Matrix A, Matrix AT, Matrix C, Matrix CT, Matrix G, Matrix GT) throws MatrixException;

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
    Matrix winogradConvolve(Matrix preprocessedFilter, Matrix A, Matrix AT, Matrix C, Matrix CT) throws MatrixException;

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
    void winogradConvolve(Matrix preprocessedFilter, Matrix result, Matrix A, Matrix AT, Matrix C, Matrix CT) throws MatrixException;

    /**
     * Calculates gradient of convolution for output.
     *
     * @param filter filter for convolutional operator.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix convolveInputGradient(Matrix filter) throws MatrixException;

    /**
     * Calculates gradient of crosscorrelation for output.
     *
     * @param filter filter for crosscorrelation operator.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix crosscorrelateInputGradient(Matrix filter) throws MatrixException;

    /**
     * Calculates gradient of convolution for input.
     *
     * @param filter filter for convolutional operator.
     * @param inputGradient input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void convolveInputGradient(Matrix filter, Matrix inputGradient) throws MatrixException;

    /**
     * Calculates gradient of crosscorrelation for input.
     *
     * @param filter filter for crosscorrelation operator.
     * @param inputGradient input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void crosscorrelateInputGradient(Matrix filter, Matrix inputGradient) throws MatrixException;

    /**
     * Calculates gradient of convolution for filter.
     *
     * @param input input for convolutional operator.
     * @return filter gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix convolveFilterGradient(Matrix input) throws MatrixException;

    /**
     * Calculates gradient of crosscorrelation for filter.
     *
     * @param input input for crosscorrelation operator.
     * @return filter gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix crosscorrelateFilterGradient(Matrix input) throws MatrixException;

    /**
     * Calculates gradient of convolution for filter.
     *
     * @param input input for convolutional operator.
     * @param filterGradient filter gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void convolveFilterGradient(Matrix input, Matrix filterGradient) throws MatrixException;

    /**
     * Calculates gradient of crosscorrelation for filter.
     *
     * @param input input for crosscorrelation operator.
     * @param filterGradient filter gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void crosscorrelateFilterGradient(Matrix input, Matrix filterGradient) throws MatrixException;

    /**
     * Calculates max pooling operation for this matrix.
     *
     * @param maxPos maximum positions for each row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix maxPool(HashMap<Integer, Integer> maxPos) throws MatrixException;

    /**
     * Calculates max pooling operation for this matrix and returns max arguments.
     *
     * @param result result matrix.
     * @param maxPos maximum positions for each row and col value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void maxPool(Matrix result, HashMap<Integer, Integer> maxPos) throws MatrixException;

    /**
     * Calculates gradient of max pooling operation for this matrix.
     *
     * @param maxPos maximum positions for each row and col value.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix maxPoolGradient(HashMap<Integer, Integer> maxPos) throws MatrixException;

    /**
     * Calculates gradient for max pool operation.
     *
     * @param inputGradient input gradient.
     * @param maxPos maximum positions for each row and col value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void maxPoolGradient(Matrix inputGradient, HashMap<Integer, Integer> maxPos) throws MatrixException;

    /**
     * Calculates random pooling operation for this matrix.
     *
     * @param inputPos input positions for each row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix randomPool(HashMap<Integer, Integer> inputPos) throws MatrixException;

    /**
     * Calculates random pooling operation for this matrix and returns max arguments.
     *
     * @param result result matrix.
     * @param inputPos input positions for each row and col value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void randomPool(Matrix result, HashMap<Integer, Integer> inputPos) throws MatrixException;

    /**
     * Calculates gradient of random pooling operation for this matrix.
     *
     * @param inputPos input positions for each row and col value.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix randomPoolGradient(HashMap<Integer, Integer> inputPos) throws MatrixException;

    /**
     * Calculates gradient for random pool operation.
     *
     * @param inputGradient input gradient.
     * @param inputPos input positions for each row and col value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void randomPoolGradient(Matrix inputGradient, HashMap<Integer, Integer> inputPos) throws MatrixException;

    /**
     * Calculates cyclic pooling operation for this matrix.
     *
     * @param inputPos input positions for each row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix cyclicPool(HashMap<Integer, Integer> inputPos) throws MatrixException;

    /**
     * Calculates cyclic pooling operation for this matrix and returns max arguments.
     *
     * @param result result matrix.
     * @param inputPos input positions for each row and col value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void cyclicPool(Matrix result, HashMap<Integer, Integer> inputPos) throws MatrixException;

    /**
     * Calculates gradient of cyclic pooling operation for this matrix.
     *
     * @param inputPos input positions for each row and col value.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix cyclicPoolGradient(HashMap<Integer, Integer> inputPos) throws MatrixException;

    /**
     * Calculates gradient for cyclic pool operation.
     *
     * @param inputGradient input gradient.
     * @param inputPos input positions for each row and col value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void cyclicPoolGradient(Matrix inputGradient, HashMap<Integer, Integer> inputPos) throws MatrixException;

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix averagePool() throws MatrixException;

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void averagePool(Matrix result) throws MatrixException;

    /**
     * Calculates gradient of average pooling operation for this matrix.
     *
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix averagePoolGradient() throws MatrixException;

    /**
     * Calculates gradient of average pooling operation for this matrix.
     *
     * @param inputGradient input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void averagePoolGradient(Matrix inputGradient) throws MatrixException;

    /**
     * Transposes matrix.
     *
     * @return new matrix but as transposed that is with flipped rows and columns.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix transpose() throws MatrixException;

    /**
     * Classifies matrix assuming multi-label classification.
     *
     * @return classified matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix classify() throws MatrixException;

    /**
     * Classifies matrix assuming multi-label classification.
     *
     * @param multiLabelThreshold threshold value for multi label classification
     * @return classified matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix classify(double multiLabelThreshold) throws MatrixException;

    /**
     * Encodes bit column vector value
     *
     * @return value
     * @throws MatrixException throws exception if matrix is not bit column vector.
     */
    int encodeToValue() throws MatrixException;

    /**
     * Splits matrix at defined position. If splitVertical is true splits vertically otherwise horizontally.
     *
     * @param position position of split
     * @param splitVertically if true splits vertically otherwise horizontally.
     * @return splitted matrix as JMatrix.
     * @throws MatrixException throws matrix exception if splitting fails.
     *
     */
    Matrix split(int position, boolean splitVertically) throws MatrixException;

    /**
     * Concatenates this and other matrix vertically (not in place and not after).
     *
     * @param other matrix to be concatenated to the end of this matrix vertically.
     * @return concatenated matrix.
     * @throws MatrixException throws exception if column dimensions of this and other matrix are not matching.
     */
    Matrix concatenateVertical(Matrix other) throws MatrixException;

    /**
     * Concatenates this matrix and other value vertically (not in place and not after).
     *
     * @param other matrix to be concatenated to the end of this matrix vertically.
     * @return concatenated matrix.
     * @throws MatrixException throws exception if column dimensions of this and other matrix are not matching.
     */
    Matrix concatenateVertical(double other) throws MatrixException;

    /**
     * Concatenates this and other matrix vertically.
     *
     * @param other matrix to be concatenated to the end of this matrix vertically.
     * @param inplace if true other matrix is concatenated to this matrix in place.
     * @param concatenateAfter if true data of other matrix is concatenated after this matrix otherwise opposite is true.
     * @return concatenated matrix.
     * @throws MatrixException throws exception if column dimensions of this and other matrix are not matching.
     */
    Matrix concatenateVertical(Matrix other, boolean inplace, boolean concatenateAfter) throws MatrixException;

    /**
     * Concatenates this matrix and other value vertically.
     *
     * @param other matrix to be concatenated to the end of this matrix vertically.
     * @param inplace if true other matrix is concatenated to this matrix in place.
     * @param concatenateAfter if true data of other matrix is concatenated after this matrix otherwise opposite is true.
     * @return concatenated matrix.
     * @throws MatrixException throws exception if column dimensions of this and other matrix are not matching.
     */
    Matrix concatenateVertical(double other, boolean inplace, boolean concatenateAfter) throws MatrixException;

    /**
     * Concatenates this and other matrix horizontally (not in place and not after).
     *
     * @param other matrix to be concatenated to the end of this matrix horizontally.
     * @return concatenated matrix.
     * @throws MatrixException throws exception if row dimensions of this and other matrix are not matching.
     */
    Matrix concatenateHorizontal(Matrix other) throws MatrixException;

    /**
     * Concatenates this matrix and other value horizontally (not in place and not after).
     *
     * @param other matrix to be concatenated to the end of this matrix horizontally.
     * @return concatenated matrix.
     * @throws MatrixException throws exception if row dimensions of this and other matrix are not matching.
     */
    Matrix concatenateHorizontal(double other) throws MatrixException;

    /**
     * Concatenates this and other matrix horizontally.
     *
     * @param other matrix to be concatenated to the end of this matrix horizontally.
     * @param inplace if true other matrix is concatenated to this matrix in place.
     * @param concatenateAfter if true data of other matrix is concatenated after this matrix otherwise opposite is true.
     * @return concatenated matrix.
     * @throws MatrixException throws exception if row dimensions of this and other matrix are not matching.
     */
    Matrix concatenateHorizontal(Matrix other, boolean inplace, boolean concatenateAfter) throws MatrixException;

    /**
     * Concatenates this matrix and other value horizontally.
     *
     * @param other matrix to be concatenated to the end of this matrix horizontally.
     * @param inplace if true other matrix is concatenated to this matrix in place.
     * @param concatenateAfter if true data of other matrix is concatenated after this matrix otherwise opposite is true.
     * @return concatenated matrix.
     * @throws MatrixException throws exception if row dimensions of this and other matrix are not matching.
     */
    Matrix concatenateHorizontal(double other, boolean inplace, boolean concatenateAfter) throws MatrixException;

    /**
     * Prints matrix in row and column format.
     *
     */
    void print();

    /**
     * Prints size (rows x columns) of matrix.
     *
     */
    void printSize();

    /**
     * Sets mask to this matrix.
     *
     * @param newMask new mask as input.
     * @throws MatrixException throws exception if new mask dimensions or mask type are not matching with this mask.
     */
    void setMask(Mask newMask) throws MatrixException;

    /**
     * Sets mask to this matrix.
     *
     */
    void setMask();

    /**
     * Removes mask from this matrix.
     *
     */
    void unsetMask();

    /**
     * Returns mask of this matrix.
     *
     * @return mask of this matrix.
     */
    Mask getMask();

    /**
     * Returns if matrix has mask at specific position.
     *
     * @param row specific row.
     * @param column specific column.
     * @return if true mask exists and is masked at specific position (row + column).
     */
    boolean hasMaskAt(int row, int column);

}
