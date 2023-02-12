/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.*;

/**
 * Implements matrix binary operation.
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
     * If true is applied as function otherwise as gradient.
     *
     */
    private boolean asFunction;

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
    public Matrix applyFunction(Matrix first, Matrix second, Matrix result) throws MatrixException {
        this.first = first;
        this.second = second;
        this.result = result;
        asFunction = true;
        switch (binaryFunctionType) {
            case DIRECT_GRADIENT -> result.setEqualTo(second);
            case POLICY_VALUE -> {
                int rows = second.getRows();
                for (int row = 0; row < rows; row++) {
                    result.setValue(row, 0 , row == 0 ? (0.5 * Math.pow(second.getValue(0, 0) - first.getValue(0, 0), 2)) : second.getValue(row, 0));
                }
            }
            // https://math.stackexchange.com/questions/1923613/partial-derivative-of-cosine-similarity
            case COS_SIM -> {
                double norm_output = first.norm(2);
                double norm_target = second.norm(2);
                first.multiply(second).divide(norm_output * norm_target, result);
            }
            default -> applyMatrixOperation();
        }
        return result;
    }

    /**
     * Calculates gradient.
     *
     * @param first first matrix.
     * @param second second matrix.
     * @return input gradient
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyGradient(Matrix first, Matrix second) throws MatrixException {
        this.first = first;
        this.second = second;
        asFunction = false;
        switch (binaryFunctionType) {
            case DIRECT_GRADIENT -> {
                return result = second;
            }
            case POLICY_VALUE -> {
                Matrix gradient = second.getNewMatrix();
                int rows = second.getRows();
                for (int row = 0; row < rows; row++) {
                    gradient.setValue(row, 0 , row == 0 ? (first.getValue(0, 0) - second.getValue(0, 0)) : second.getValue(row, 0));
                }
                return result = gradient;
            }
            // https://math.stackexchange.com/questions/1923613/partial-derivative-of-cosine-similarity
            case COS_SIM -> {
                double norm_output = first.norm(2);
                double norm_target = second.norm(2);
                double norm_multiply = norm_output * norm_target;
                Matrix cos_sim = first.multiply(second).divide(norm_multiply);
                return result = first.divide(norm_multiply).subtract(second.divide(Math.pow(norm_output, 2)).multiply(cos_sim));
            }
            default -> {
                result = first.getNewMatrix();
                applyMatrixOperation();
                return result;
            }
        }
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
        return result = outputGradient.multiply(applyGradient(first, second));
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
        result.setValue(row, column, asFunction ? matrixBinaryOperation.execute(value, second.getValue(row, column)) : matrixGradientBinaryOperation.execute(value, second.getValue(row, column)));
    }

}
