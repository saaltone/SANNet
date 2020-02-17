/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import utils.matrix.BinaryFunction;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;

/**
 * Class that describes expression for binary function.
 *
 */
public class BinaryFunctionExpression extends AbstractBinaryExpression implements Serializable {

    /**
     * BinaryFunction used.
     *
     */
    private final BinaryFunction binaryFunction;

    /**
     * Lambda operation to calculate binaryFunction.
     *
     */
    private final Matrix.MatrixBinaryOperation binaryFunctionPrime;

    /**
     * Lambda operation to calculate derivative of binaryFunction.
     *
     */
    private final Matrix.MatrixBinaryOperation binaryDerivativePrime;

    /**
     * Constructor for binary function.
     *
     * @param expressionID unique ID for expression.
     * @param arg1 first argument.
     * @param arg2 second argument.
     * @param result result of expression.
     * @param binaryFunction BinaryFunction.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public BinaryFunctionExpression(int expressionID, Node arg1, Node arg2, Node result, BinaryFunction binaryFunction) throws MatrixException {
        super(expressionID, arg1, arg2, result);
        this.binaryFunction = binaryFunction;
        this.binaryFunctionPrime = null;
        this.binaryDerivativePrime = null;
    }

    /**
     * Constructor for expression.
     *
     * @param expressionID unique ID for expression.
     * @param arg1 first argument.
     * @param arg2 second argument.
     * @param result result of expression.
     * @param binaryFunctionPrime prime matrix binaryFunction.
     * @param binaryDerivativePrime prime matrix binaryFunction derivative.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public BinaryFunctionExpression(int expressionID, Node arg1, Node arg2, Node result, Matrix.MatrixBinaryOperation binaryFunctionPrime, Matrix.MatrixBinaryOperation binaryDerivativePrime) throws MatrixException {
        super(expressionID, arg1, arg2, result);
        this.binaryFunction = null;
        this.binaryFunctionPrime = binaryFunctionPrime;
        this.binaryDerivativePrime = binaryDerivativePrime;
    }

    /**
     * Returns BinaryFunction of expression.
     *
     * @return BinaryFunction of expression.
     */
    public BinaryFunction getBinaryFunction() {
        return binaryFunction;
    }

    /**
     * Returns BinaryFunction prime.
     *
     * @return BinaryFunction prime.
     */
    public Matrix.MatrixBinaryOperation getBinaryFunctionPrime() {
        return binaryFunctionPrime;
    }

    /**
     * Returns BinaryFunction derivative prime.
     *
     * @return BinaryFunction derivative prime.
     */
    public Matrix.MatrixBinaryOperation getBinaryDerivativePrime() {
        return binaryDerivativePrime;
    }

    /**
     * Calculates expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int index) throws MatrixException {
        if (arg1.getMatrix(index) == null || arg2.getMatrix(index) == null) throw new MatrixException("Arguments for bi operation not defined");
        if (binaryFunction != null) result.setMatrix(index, binaryFunction.applyFunction(arg1.getMatrix(index), arg2.getMatrix(index)));
        else result.setMatrix(index, arg1.getMatrix(index).applyBi(arg2.getMatrix(index), binaryFunctionPrime));
    }

    /**
     * Calculates gradient of expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int index) throws MatrixException {
        if (result.getGradient(index) == null) throw new MatrixException("Result gradient not defined.");
        Matrix inBiGradient;
        if (binaryFunction != null) inBiGradient = binaryFunction.applyGradient(result.getMatrix(index), arg2.getMatrix(index), result.getGradient(index));
        else inBiGradient = result.getGradient(index).multiply(result.getMatrix(index).applyBi(arg2.getMatrix(index), binaryDerivativePrime));
        arg1.updateGradient(index, inBiGradient, true);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        System.out.print("BINARYFUN: " + " " + arg1 + " " + arg2 + " " + result);
    }

}
