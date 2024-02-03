/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.matrix;

import utils.configurable.DynamicParamException;
import utils.procedure.ProcedureFactory;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Interface that defines matrix with extensive set of matrix operations and masking for matrix.<br>
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
     * Returns sub-matrices within matrix.
     *
     * @return sub-matrices within matrix.
     */
    ArrayList<Matrix> getSubMatrices();

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
     * @param depth depth of value to be set.
     * @param value new value to be set.
     */
    void setValue(int row, int column, int depth, double value);

    /**
     * Returns value of matrix at specific row and column.
     *
     * @param row row of value to be returned.
     * @param column column of value to be returned.
     * @param depth depth of value to be returned.
     * @return value of row and column.
     */
    double getValue(int row, int column, int depth);

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
     * Returns total depth defined for matrix.
     *
     * @return total depth defined for matrix.
     */
    int getTotalDepth();

    /**
     * Returns size (rows * columns * depth) of matrix effectively with slicing considered.<br>
     *
     * @return size (rows * columns * depth) of matrix effectively with slicing considered.
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
     * Returns depth of matrix effectively with slicing considered.<br>
     *
     * @return depth of matrix effectively with slicing considered.
     */
    int getDepth();

    /**
     * Returns matrix of given size (rows x columns)
     *
     * @param rows rows
     * @param columns columns
     * @param depth depth
     * @return new matrix
     * @throws MatrixException throws exception if new mask dimensions or mask type are not matching with this mask.
     */
    Matrix getNewMatrix(int rows, int columns, int depth) throws MatrixException;

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
     * @throws MatrixException throws exception if new mask dimensions or mask type are not matching with this mask.
     */
    Matrix getNewMatrix(boolean asTransposed) throws MatrixException;

    /**
     * Returns constant matrix
     *
     * @param constant constant
     * @return new matrix
     */
    Matrix getNewMatrix(double constant);

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
     * @return newly created copy of matrix.
     * @throws MatrixException throws exception if mask is not set or cloning of matrix fails.
     */
    Matrix copy() throws MatrixException;

    /**
     * Creates new matrix with object full copy of this matrix.
     *
     * @param canBeSliced if true matrix can be slides otherwise cannot be sliced.
     * @return newly created copy of matrix.
     * @throws MatrixException throws exception if mask is not set or cloning of matrix fails.
     */
    Matrix copy(boolean canBeSliced) throws MatrixException;

    /**
     * Redimensions matrix assuming new dimensions are matching.
     *
     * @param newRows new row size
     * @param newColumns new column size
     * @param newDepth new depth size.
     * @return redimensioned matrix.
     * @throws MatrixException throws exception if redimensioning fails.
     */
    Matrix redimension(int newRows, int newColumns, int newDepth) throws MatrixException;

    /**
     * Redimensions matrix assuming new dimensions are matching.
     *
     * @param newRows new row size
     * @param newColumns new column size
     * @param newDepth new depth size.
     * @param copyData if true matrix data is copied and if false referenced.
     * @return redimensioned matrix.
     * @throws MatrixException throws exception if redimensioning fails.
     */
    Matrix redimension(int newRows, int newColumns, int newDepth, boolean copyData) throws MatrixException;

    /**
     * Slices matrix.
     *
     * @param startRow start row of slice.
     * @param startColumn start column of slice.
     * @param startDepth start depth of slice.
     * @param endRow  end row of slice.
     * @param endColumn  end column of slice.
     * @param endDepth  end depth of slice.
     * @throws MatrixException throws exception if slicing fails.
     */
    void slice(int startRow, int startColumn, int startDepth, int endRow, int endColumn, int endDepth) throws MatrixException;

    /**
     * Removes slicing of matrix.
     *
     * @throws MatrixException throws exception if matrix cannot be sliced.
     */
    void unslice() throws MatrixException;

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
     * @param unaryFunction unary function.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    Matrix apply(UnaryFunction unaryFunction, boolean inplace) throws MatrixException;

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
     * @param unaryFunctionType unaryFunction type to be applied.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return matrix which stores operation result.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix apply(UnaryFunctionType unaryFunctionType, boolean inplace) throws MatrixException, DynamicParamException;

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
     * @param other other matrix
     * @param binaryFunction binaryFunction to be applied.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return matrix which stores operation result.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix applyBi(Matrix other, BinaryFunction binaryFunction, boolean inplace) throws MatrixException;

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
     * @param other              other matrix
     * @param binaryFunctionType binaryFunction type to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException       throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix applyBi(Matrix other, BinaryFunctionType binaryFunctionType) throws MatrixException, DynamicParamException;

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
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    Matrix add(double constant) throws MatrixException;

    /**
     * Adds this matrix by other matrix.
     *
     * @param other other matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    void addBy(Matrix other) throws MatrixException;

    /**
     * Adds this matrix by constant.
     *
     * @param constant constant.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    void addBy(double constant) throws MatrixException;

    /**
     * Increment value of specific row, column and depth.
     *
     * @param row row of value to be added.
     * @param column column of value to be added.
     * @param depth depth of value to be added.
     * @param value to be added.
     */
    void addByValue(int row, int column, int depth, double value);

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
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    Matrix subtract(double constant) throws MatrixException;

    /**
     * Subtracts this matrix by other matrix.
     *
     * @param other other matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    void subtractBy(Matrix other) throws MatrixException;

    /**
     * Subtracts this matrix by constant.
     *
     * @param constant constant.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    void subtractBy(double constant) throws MatrixException;

    /**
     * Decrease value of specific row, column and depth.
     *
     * @param row row of value to be decreased.
     * @param column column of value to be decreased.
     * @param depth depth of value to be decreased.
     * @param value to be decreased.
     */
    void subtractByValue(int row, int column, int depth, double value);

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
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    Matrix multiply(double constant) throws MatrixException;

    /**
     * Multiplies this matrix by other matrix.
     *
     * @param other other matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    void multiplyBy(Matrix other) throws MatrixException;

    /**
     * Multiplies this matrix by constant.
     *
     * @param constant constant.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    void multiplyBy(double constant) throws MatrixException;

    /**
     * Multiply value of specific row, column and depth.
     *
     * @param row row of value to be multiplied.
     * @param column column of value to be multiplied.
     * @param depth depth of value to be multiplied.
     * @param value to be multiplied.
     */
    void multiplyByValue(int row, int column, int depth, double value);

    /**
     * Divides this matrix element wise with other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    Matrix divide(Matrix other) throws MatrixException;

    /**
     * Divides this matrix element wise with constant.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param constant constant used as divider value.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    Matrix divide(double constant) throws MatrixException;

    /**
     * Divides this matrix by other matrix.
     *
     * @param other other matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    void divideBy(Matrix other) throws MatrixException;

    /**
     * Divides this matrix by constant.
     *
     * @param constant constant.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    void divideBy(double constant) throws MatrixException;

    /**
     * Divide value of specific row, column and depth.
     *
     * @param row row of value to be divided.
     * @param column column of value to be divided.
     * @param depth depth of value to be divided.
     * @param value to be divided.
     */
    void divideByValue(int row, int column, int depth, double value);

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
     * Raises this matrix element wise to the power of value power.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param power value of power.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix power(double power) throws MatrixException, DynamicParamException;

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
     * Takes element wise max value of this and other value.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param other value which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix max(double other) throws MatrixException, DynamicParamException;

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
     * Takes element wise min value of this and other matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param other value which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix min(double other) throws MatrixException, DynamicParamException;

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
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return sum of matrix.
     * @throws MatrixException not thrown in any situation.
     */
    Matrix sumAsMatrix(int direction) throws MatrixException;

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
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @throws MatrixException not thrown in any situation.
     * @return mean of matrix.
     */
    Matrix meanAsMatrix(int direction) throws MatrixException;

    /**
     * Takes variance of elements of this matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @return variance of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    double variance() throws MatrixException, DynamicParamException;

    /**
     * Takes variance of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return variance of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    double variance(double mean) throws MatrixException, DynamicParamException;

    /**
     * Takes variance of elements of this matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return variance of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix varianceAsMatrix(int direction) throws MatrixException, DynamicParamException;

    /**
     * Takes variance of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean      mean value given as input.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return variance of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix varianceAsMatrix(Matrix mean, int direction) throws MatrixException, DynamicParamException;

    /**
     * Takes standard deviation of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return standard deviation of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    double standardDeviation() throws MatrixException, DynamicParamException;

    /**
     * Takes standard deviation of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return standard deviation of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    double standardDeviation(double mean) throws MatrixException, DynamicParamException;

    /**
     * Takes standard deviation of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return standard deviation of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix standardDeviationAsMatrix(int direction) throws MatrixException, DynamicParamException;

    /**
     * Takes standard deviation of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean      mean value given as input.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @return standard deviation of matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix standardDeviationAsMatrix(Matrix mean, int direction) throws MatrixException, DynamicParamException;

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
     * Normalizes matrix by removing mean and variance.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @param inplace if true matrix is normalized in place otherwise copy of normalized matrix is returned.
     * @return normalized matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix normalize(boolean inplace) throws MatrixException, DynamicParamException;

    /**
     * Calculates exponential moving average.
     *
     * @param currentAverage current average value
     * @param momentum degree of weighting decrease for exponential moving average.
     * @return updated average with new average value included.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix exponentialMovingAverage(Matrix currentAverage, double momentum) throws MatrixException;

    /**
     * Calculates cumulative moving average CMAn = CMAn-1 + (currentAverage - CMAn-1) / sampleCount
     *
     * @param currentMovingAverage current cumulative moving average
     * @param sampleCount current sample count
     * @return updated cumulative moving average.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix cumulativeMovingAverage(Matrix currentMovingAverage, int sampleCount) throws MatrixException;

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
    Matrix dropout(double probability, boolean monte_carlo, boolean inplace) throws MatrixException;

    /**
     * Clips gradient matrix against threshold.
     *
     * @param threshold threshold.
     * @return clipped gradient matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix gradientClip(double threshold) throws MatrixException;

    /**
     * Implements matrix noising.
     *
     * @param noise noise
     * @param inplace if true clipping in done in place otherwise not.
     * @return result of drop out.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix noise(double noise, boolean inplace) throws MatrixException;

    /**
     * Returns softmax of this matrix.
     *
     * @return softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix softmax() throws MatrixException, DynamicParamException;

    /**
     * Returns softmax of this matrix.
     *
     * @param softmaxTau tau value for Softmax.
     * @return softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix softmax(double softmaxTau) throws MatrixException, DynamicParamException;

    /**
     * Returns Gumbel softmax of this matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @return Gumbel softmax matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix gumbelSoftmax() throws MatrixException, DynamicParamException;

    /**
     * Returns Gumbel softmax of this matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @param softmaxTau tau value for Softmax.
     * @return Gumbel softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix gumbelSoftmax(double softmaxTau) throws MatrixException, DynamicParamException;

    /**
     * Transposes matrix.
     *
     * @return new matrix but as transposed that is with flipped rows and columns.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix transpose() throws MatrixException, DynamicParamException;

    /**
     * Checks if matrix is transposed.
     *
     * @return true is matrix is transposed otherwise false.
     */
    boolean isTransposed();

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
     * Splits matrix at defined position. If splitVertical is true splits vertically otherwise horizontally.
     *
     * @param position position of split
     * @param splitVertically if true splits vertically otherwise horizontally.
     * @return split matrix as JMatrix.
     * @throws MatrixException throws matrix exception if splitting fails.
     *
     */
    Matrix split(int position, boolean splitVertically) throws MatrixException;

    /**
     * Joins two matrices either vertically or horizontally.
     *
     * @param other other matrix
     * @param joinedVertically if true joined vertically otherwise horizontally
     * @return joined matrix
     * @throws MatrixException throws matrix exception if joining fails.
     */
    Matrix join(Matrix other, boolean joinedVertically) throws MatrixException;

    /**
     * Unjoins matrix at specific row and column.
     *
     * @param unjoinAtRow unjoins at row.
     * @return result matrix.
     * @throws MatrixException throws matrix exception if unjoining fails.
     */
    Matrix unjoin(int unjoinAtRow) throws MatrixException;

    /**
     * Unjoins matrix at specific row and column.
     *
     * @param unjoinAtRow unjoins at row.
     * @param unjoinAtColumn unjoins at column.
     * @param unjoinAtDepth unjoins at depth.
     * @param unjoinRows unjoins specific number of rows.
     * @param unjoinColumns unjoins specific number of column.
     * @param unjoinDepth unjoins specific depth.
     * @return result matrix.
     * @throws MatrixException throws matrix exception if unjoining fails.
     */
    Matrix unjoin(int unjoinAtRow, int unjoinAtColumn, int unjoinAtDepth, int unjoinRows, int unjoinColumns, int unjoinDepth) throws MatrixException;

    /**
     * Flattens matrix into one dimensional column vector (matrix)
     *
     * @return flattened matrix
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix flatten() throws MatrixException;

    /**
     * Returns unflattened matrix i.e. samples that have been unflattened from single column vector.
     *
     * @param rows rows of unflattened matrix.
     * @param columns columns of unflattened matrix.
     * @param depth depth of unflattened matrix.
     * @return unflattened matrix.
     * @throws MatrixException throws matrix exception if joining fails.
     */
    Matrix unflatten(int rows, int columns, int depth) throws MatrixException;

    /**
     * Encodes bit column vector value
     *
     * @return value
     * @throws MatrixException throws exception if matrix is not bit column vector.
     */
    int encodeBitColumnVectorToValue() throws MatrixException;

    /**
     * Returns multinomial distribution. Assumes single trial.
     *
     * @return multinomial distribution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix getMultinomial() throws MatrixException;

    /**
     * Returns multinomial distribution.
     *
     * @param numberOfTrials number of trials.
     * @return multinomial distribution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix getMultinomial(int numberOfTrials) throws MatrixException;

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
     * Sets filter depth.
     *
     * @param filterDepth filter depth.
     */
    void setFilterDepth(int filterDepth);

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
     * Returns filter depth.
     *
     * @return filter depth.
     */
    int getFilterDepth();

    /**
     * Sets if convolution is depth separable.
     *
     * @param isDepthSeparable is true convolution is depth separable.
     */
    void setIsDepthSeparable(boolean isDepthSeparable);

    /**
     * Returns if convolution is depth separable.
     *
     * @return if true convolution is depth separable.
     */
    boolean getIsDepthSeparable();

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated value of convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix convolve(Matrix filter) throws MatrixException;

    /**
     * Calculates crosscorrelation between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated value of crosscorrelation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix crosscorrelate(Matrix filter) throws MatrixException;

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return calculated value of Winograd convolution.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Matrix winogradConvolve(Matrix filter) throws MatrixException, DynamicParamException;

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
    Matrix winogradConvolve(Matrix filter, Matrix A, Matrix AT, Matrix C, Matrix CT, Matrix G, Matrix GT) throws MatrixException, DynamicParamException;

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
    Matrix winogradConvolve(Matrix preprocessedFilter, Matrix A, Matrix AT, Matrix C, Matrix CT) throws MatrixException, DynamicParamException;

    /**
     * Calculates max pooling operation for this matrix.
     *
     * @param maxPos maximum positions for each row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix maxPool(HashMap<Integer, Integer> maxPos) throws MatrixException;

    /**
     * Calculates random pooling operation for this matrix.
     *
     * @param inputPos input positions for each row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix randomPool(HashMap<Integer, Integer> inputPos) throws MatrixException;

    /**
     * Calculates cyclic pooling operation for this matrix.
     *
     * @param inputPos input positions for each row and col value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix cyclicPool(HashMap<Integer, Integer> inputPos) throws MatrixException;

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix averagePool() throws MatrixException;

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
     * Returns if matrix is masked at specific position.
     *
     * @param row specific row.
     * @param column specific column.
     * @param depth specific depth.
     * @return if true mask exists and is masked at specific position (row + column).
     */
    boolean hasMaskAt(int row, int column, int depth);

}
