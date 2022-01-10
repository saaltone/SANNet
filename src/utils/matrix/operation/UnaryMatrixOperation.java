/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunction;
import utils.matrix.UnaryFunctionType;

/**
 * Defines matrix unary operation.
 *
 */
public class UnaryMatrixOperation extends AbstractMatrixOperation {

    /**
     * First matrix.
     *
     */
    private Matrix first;

    /**
     * Result matrix.
     *
     */
    private Matrix result;

    /**
     * Matrix unary function.
     *
     */
    private final UnaryFunction unaryFunction;

    /**
     * Matrix unary function type
     *
     */
    private final UnaryFunctionType unaryFunctionType;

    /**
     * Matrix unary function type
     *
     */
    private final Matrix.MatrixUnaryOperation matrixUnaryOperation;

    /**
     * Matrix unary function type
     *
     */
    private final Matrix.MatrixUnaryOperation matrixGradientUnaryOperation;

    /**
     * Constructor for matrix unary operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param matrixUnaryOperation matrix unary operation.
     */
    public UnaryMatrixOperation(int rows, int columns, Matrix.MatrixUnaryOperation matrixUnaryOperation) {
        super(rows, columns, true);
        this.unaryFunction = null;
        this.unaryFunctionType = null;
        this.matrixUnaryOperation = matrixUnaryOperation;
        this.matrixGradientUnaryOperation = null;
    }

    /**
     * Constructor for matrix unary operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param unaryFunction unary function.
     */
    public UnaryMatrixOperation(int rows, int columns, UnaryFunction unaryFunction) {
        super(rows, columns, true);
        this.unaryFunction = unaryFunction;
        this.unaryFunctionType = unaryFunction.getType();
        this.matrixUnaryOperation = unaryFunction.getFunction();
        this.matrixGradientUnaryOperation = unaryFunction.getDerivative();
    }

    /**
     * Applies operation.
     *
     * @param first first matrix.
     * @param result result matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix first, Matrix result) throws MatrixException {
        this.first = first;
        this.result = result;
        applyMatrixOperation();
        return result;
    }

    /**
     * Applies function to matrix (value)
     *
     * @param first first matrix
     * @param result result matrix
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyFunction(Matrix first, Matrix result) throws MatrixException {
        this.first = first;
        this.result = result;
        switch (unaryFunctionType) {
            case SOFTMAX -> first.softmax(result);
            case GUMBEL_SOFTMAX -> first.gumbelSoftmax(result, unaryFunction.getGumbelSoftmaxTau());
            default -> first.apply(result, matrixUnaryOperation);
        }
    }

    /**
     * Calculates inner gradient.
     *
     * @param first first matrix.
     * @param outputGradient output gradient.
     * @return input gradient
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyGradient(Matrix first, Matrix outputGradient) throws MatrixException {
        return switch (unaryFunctionType) {
            case SOFTMAX, GUMBEL_SOFTMAX -> first.softmaxGrad().dot(outputGradient);
            default -> outputGradient.multiply(first.apply(matrixGradientUnaryOperation));
        };
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
        return null;
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     */
    public void apply(int row, int column, double value) {
        result.setValue(row, column, matrixUnaryOperation.execute(value));
    }

}
