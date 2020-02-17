/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunction;

import java.io.Serializable;

/**
 * Class that describes expression for unary function.
 *
 */
public class UnaryFunctionExpression extends AbstractUnaryExpression implements Serializable {

    /**
     * UnaryFunction used.
     *
     */
    private final UnaryFunction unaryFunction;

    /**
     * Lambda operation to calculate unaryFunction.
     *
     */
    private final Matrix.MatrixUnaryOperation unaryFunctionPrime;

    /**
     * Lambda operation to calculate derivative of unaryFunction.
     *
     */
    private final Matrix.MatrixUnaryOperation uniDerivativePrime;

    /**
     * Constructor for unary function.
     *
     * @param expressionID unique ID for expression.
     * @param arg1 first argument.
     * @param result result.
     * @param unaryFunction UnaryFunction.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    UnaryFunctionExpression(int expressionID, Node arg1, Node result, UnaryFunction unaryFunction) throws MatrixException {
        super(expressionID, arg1, result);
        this.unaryFunction = unaryFunction;
        this.unaryFunctionPrime = null;
        this.uniDerivativePrime = null;
    }

    /**
     * Constructor for expression.
     *
     * @param expressionID unique ID for expression.
     * @param arg1 first argument.
     * @param result result for expression.
     * @param unaryFunctionPrime prime matrix UnaryFunction.
     * @param uniDerivativePrime prime matrix UnaryFunction derivative.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public UnaryFunctionExpression(int expressionID, Node arg1, Node result, Matrix.MatrixUnaryOperation unaryFunctionPrime, Matrix.MatrixUnaryOperation uniDerivativePrime) throws MatrixException {
        super(expressionID, arg1, result);
        this.unaryFunction = null;
        this.unaryFunctionPrime = unaryFunctionPrime;
        this.uniDerivativePrime = uniDerivativePrime;
    }

    /**
     * Returns UnaryFunction of expression.
     *
     * @return UnaryFunction of expression.
     */
    public UnaryFunction getUnaryFunction() {
        return unaryFunction;
    }

    /**
     * Returns UnaryFunction prime.
     *
     * @return UnaryFunction prime.
     */
    public Matrix.MatrixUnaryOperation getUnaryFunctionPrime() {
        return unaryFunctionPrime;
    }

    /**
     * Returns UnaryFunction derivative prime.
     *
     * @return UnaryFunction derivative prime.
     */
    public Matrix.MatrixUnaryOperation getUnaryDerivativePrime() {
        return uniDerivativePrime;
    }

    /**
     * Calculates expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int index) throws MatrixException {
        if (arg1.getMatrix(index) == null) throw new MatrixException("Argument for unary operation not defined");
        if (unaryFunction != null) result.setMatrix(index, unaryFunction.applyFunction(arg1.getMatrix(index)));
        else result.setMatrix(index, arg1.getMatrix(index).apply(unaryFunctionPrime));
    }

    /**
     * Calculates gradient of expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int index) throws MatrixException {
        if (result.getGradient(index) == null) throw new MatrixException("Result gradient not defined.");
        Matrix inUniGradient;
        if (unaryFunction != null) inUniGradient = unaryFunction.applyGradient(result.getMatrix(index), result.getGradient(index));
        else inUniGradient = result.getGradient(index).multiply(result.getMatrix(index).apply(uniDerivativePrime));
        arg1.updateGradient(index, inUniGradient, true);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        System.out.print("UNARYFUN: " + " " + arg1 + " " + result);
    }

}
