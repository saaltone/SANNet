/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunction;
import utils.matrix.UnaryFunctionType;

/**
 * Implements matrix unary operation.
 *
 */
public class UnaryMatrixOperation extends AbstractMatrixOperation {

    /**
     * First matrix.
     *
     */
    private transient Matrix first;

    /**
     * Result matrix.
     *
     */
    private transient Matrix result;

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
     * If true is applied as function otherwise as gradient.
     *
     */
    private transient boolean asFunction;

    /**
     * Constructor for matrix unary operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param unaryFunction unary function.
     */
    public UnaryMatrixOperation(int rows, int columns, int depth, UnaryFunction unaryFunction) {
        super(rows, columns, depth, true);
        this.unaryFunction = unaryFunction;
        this.unaryFunctionType = unaryFunction.getType();
        this.matrixUnaryOperation = unaryFunction.getFunction();
        this.matrixGradientUnaryOperation = unaryFunction.getDerivative();
    }

    /**
     * Applies operation.
     *
     * @param first first matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyFunction(Matrix first) throws MatrixException {
        return applyFunction(first, false);
    }

    /**
     * Applies operation.
     *
     * @param first first matrix.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyFunction(Matrix first, boolean inplace) throws MatrixException {
        this.first = first;
        asFunction = true;
        switch (unaryFunctionType) {
            case SOFTMAX -> result = first.softmax(unaryFunction.getSoftmaxTau());
            case GUMBEL_SOFTMAX -> result = first.gumbelSoftmax(unaryFunction.getSoftmaxTau());
            case TRANSPOSE -> {
                result = first.getNewMatrix(first.getColumns(), first.getRows(), getDepth());
                applyMatrixOperation();
            }
            default -> {
                result = inplace ? first : first.getNewMatrix(getRows(), getColumns(), getDepth());
                applyMatrixOperation();
            }
        }
        return result;
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
        this.first = first;
        asFunction = false;
        switch (unaryFunctionType) {
            case SOFTMAX, GUMBEL_SOFTMAX -> {
                return result = first.softmaxGrad().dot(outputGradient);
            }
            case TRANSPOSE -> {
                return result = outputGradient.transpose();
            }
            default -> {
                result = first.getNewMatrix(getRows(), getColumns(), getDepth());
                applyMatrixOperation();
                return result = outputGradient.multiply(result);
            }
        }
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
    public Matrix getOther() {
        return null;
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     * @param value current value.
     */
    public void apply(int row, int column, int depth, double value) {
        if (unaryFunctionType == UnaryFunctionType.TRANSPOSE) result.setValue(column, row, depth, value);
        else result.setValue(row, column, depth, asFunction ? matrixUnaryOperation.execute(value) : matrixGradientUnaryOperation.execute(value));
    }

}
