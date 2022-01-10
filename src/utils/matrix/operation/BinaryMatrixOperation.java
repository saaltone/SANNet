/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.*;

/**
 * Defines matrix binary operation.
 *
 */
public class BinaryMatrixOperation extends AbstractMatrixOperation {

    /**
     * First matrix.
     *
     */
    private Matrix first;

    /**
     * Second matrix.
     *
     */
    private Matrix second;

    /**
     * Result matrix.
     *
     */
    private Matrix result;

    /**
     * Matrix binary function.
     *
     */
    private final BinaryFunction binaryFunction;

    /**
     * Matrix binary function type
     *
     */
    private final BinaryFunctionType binaryFunctionType;

    /**
     * Matrix binary function type
     *
     */
    private final Matrix.MatrixBinaryOperation matrixBinaryOperation;

    /**
     * Matrix binary function type
     *
     */
    private final Matrix.MatrixBinaryOperation matrixGradientBinaryOperation;

    /**
     * Constructor for matrix unary operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param matrixBinaryOperation matrix binary operation.
     */
    public BinaryMatrixOperation(int rows, int columns, Matrix.MatrixBinaryOperation matrixBinaryOperation) {
        super(rows, columns, true);
        this.binaryFunction = null;
        this.binaryFunctionType = null;
        this.matrixBinaryOperation = matrixBinaryOperation;
        this.matrixGradientBinaryOperation = null;
    }

    /**
     * Constructor for matrix binary operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param binaryFunction binary function.
     */
    public BinaryMatrixOperation(int rows, int columns, BinaryFunction binaryFunction) {
        super(rows, columns, true);
        this.binaryFunction = binaryFunction;
        this.binaryFunctionType = binaryFunction.getType();
        this.matrixBinaryOperation = binaryFunction.getFunction();
        this.matrixGradientBinaryOperation = binaryFunction.getDerivative();
    }


    /**
     * Applies matrix operation.
     *
     * @param first first matrix.
     * @param second second matrix.
     * @param result result matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix first, Matrix second, Matrix result) throws MatrixException {
        this.first = first;
        this.second = second;
        this.result = result;
        applyMatrixOperation();
        return result;
    }

    /**
     * Applies function to first and second matrix.
     *
     * @param first first matrix.
     * @param second second matrix.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyFunction(Matrix first, Matrix second, Matrix result) throws MatrixException {
        first.applyBi(second, result, matrixBinaryOperation);
    }

    /**
     * Calculates gradient.
     *
     * @param first first matrix.
     * @param second second matrix.
     * @param outputGradient output gradient.
     * @return input gradient
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyGradient(Matrix first, Matrix second, Matrix outputGradient) throws MatrixException {
        return outputGradient.multiply(first.applyBi(second, matrixGradientBinaryOperation));
    }

    /**
     * Returns target matrix.
     *
     * @return target matrix.
     */
    protected Matrix getTargetMatrix() {
        return first;
    }

    /**
     * Returns another matrix used in operation.
     *
     * @return another matrix used in operation.
     */
    public Matrix getAnother() {
        return second;
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     */
    public void apply(int row, int column, double value) {
        result.setValue(row, column, matrixBinaryOperation.execute(value, second.getValue(row, column)));
    }

}
