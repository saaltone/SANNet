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
     * Second matrix.
     *
     */
    private Matrix second;

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
     * @param depth depth for operation.
     * @param binaryFunction binary function.
     */
    public BinaryMatrixOperation(int rows, int columns, int depth, BinaryFunction binaryFunction) {
        super(rows, columns, depth, true);
        this.binaryFunctionType = binaryFunction.getType();
        this.matrixBinaryOperation = binaryFunction.getFunction();
        this.matrixGradientBinaryOperation = binaryFunction.getDerivative();
    }

    /**
     * Applies matrix operation.
     *
     * @param first  first matrix.
     * @param second second matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyFunction(Matrix first, Matrix second) throws MatrixException {
        return applyFunction(first, second, false);
    }

    /**
     * Applies matrix operation.
     *
     * @param first  first matrix.
     * @param second second matrix.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyFunction(Matrix first, Matrix second, boolean inplace) throws MatrixException {
        this.second = second;
        asFunction = true;
        switch (binaryFunctionType) {
            case DIRECT_GRADIENT -> {
                return second;
            }
            case POLICY_VALUE -> {
                Matrix result = first.getNewMatrix();
                int rows = first.getRows();
                int totalDepth = first.getDepth();
                for (int depth = 0; depth < totalDepth; depth++) {
                    for (int row = 0; row < rows; row++) {
                        result.setValue(row, 0, depth, row == 0 ? (0.5 * Math.pow(second.getValue(0, 0, depth) - first.getValue(0, 0, depth), 2)) : second.getValue(row, 0, depth));
                    }
                }
                return result;
            }
            // https://math.stackexchange.com/questions/1923613/partial-derivative-of-cosine-similarity
            case COS_SIM -> {
                double norm_output = first.norm(2);
                double norm_target = second.norm(2);
                return first.multiply(second).divide(norm_output * norm_target);
            }
            default -> {
                return applyMatrixOperation(first, second, inplace ? first : !first.isScalar() ? first.getNewMatrix() : second.getNewMatrix());
            }
        }
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
        this.second = second;
        asFunction = false;
        switch (binaryFunctionType) {
            case DIRECT_GRADIENT -> {
                return second;
            }
            case POLICY_VALUE -> {
                Matrix gradient = second.getNewMatrix();
                int rows = second.getRows();
                int totalDepth = first.getDepth();
                for (int depth = 0; depth < totalDepth; depth++) {
                    for (int row = 0; row < rows; row++) {
                        gradient.setValue(row, 0, depth, row == 0 ? (first.getValue(0, 0, depth) - second.getValue(0, 0, depth)) : second.getValue(row, 0, depth));
                    }
                }
                return gradient;
            }
            // https://math.stackexchange.com/questions/1923613/partial-derivative-of-cosine-similarity
            case COS_SIM -> {
                double norm_output = first.norm(2);
                double norm_target = second.norm(2);
                double norm_multiply = norm_output * norm_target;
                Matrix cos_sim = first.multiply(second).divide(norm_multiply);
                return first.divide(norm_multiply).subtract(second.divide(Math.pow(norm_output, 2)).multiply(cos_sim));
            }
            default -> {
                return applyMatrixOperation(first, second, first.getNewMatrix());
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
        return outputGradient.multiply(applyGradient(first, second));
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
        result.setValue(row, column, depth, asFunction ? matrixBinaryOperation.execute(value, second.getValue(row, column, depth)) : matrixGradientBinaryOperation.execute(value, second.getValue(row, column, depth)));
    }

}
