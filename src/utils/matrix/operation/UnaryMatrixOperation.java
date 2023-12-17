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
     * Defines softmax operation in case type if softmax.
     *
     */
    private final SoftmaxMatrixOperation softmaxMatrixOperation;

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
        switch (unaryFunctionType) {
            case SOFTMAX -> softmaxMatrixOperation = new SoftmaxMatrixOperation(rows, columns, depth, unaryFunction.getSoftmaxTau(), false);
            case GUMBEL_SOFTMAX -> softmaxMatrixOperation = new SoftmaxMatrixOperation(rows, columns, depth, unaryFunction.getSoftmaxTau(), true);
            default -> softmaxMatrixOperation = null;
        }
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
        asFunction = true;
        switch (unaryFunctionType) {
            case SOFTMAX, GUMBEL_SOFTMAX -> {
                return softmaxMatrixOperation.applyFunction(first);
            }
            case TRANSPOSE -> {
                return applyMatrixOperation(first, null, first.getNewMatrix(first.getColumns(), first.getRows(), getDepth()));
            }
            default -> {
                return applyMatrixOperation(first, null, inplace ? first : first.getNewMatrix(getRows(), getColumns(), getDepth()));
            }
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
        asFunction = false;
        switch (unaryFunctionType) {
            case SOFTMAX, GUMBEL_SOFTMAX -> {
                return softmaxMatrixOperation.applyGradient(first, outputGradient);
            }
            case TRANSPOSE -> {
                return outputGradient.transpose();
            }
            default -> {
                return outputGradient.multiply(applyMatrixOperation(first, null, first.getNewMatrix(getRows(), getColumns(), getDepth())));
            }
        }
    }

    /**
     * Applies operation.
     *
     * @param row    current row.
     * @param column current column.
     * @param depth  current depth.
     * @param value  current value.
     * @param result result matrix.
     */
    public void apply(int row, int column, int depth, double value, Matrix result) {
        if (unaryFunctionType == UnaryFunctionType.TRANSPOSE) result.setValue(column, row, depth, value);
        else result.setValue(row, column, depth, asFunction ? matrixUnaryOperation.execute(value) : matrixGradientUnaryOperation.execute(value));
    }

}
