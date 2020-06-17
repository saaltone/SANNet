/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import utils.matrix.*;

import java.io.Serializable;

public class NormExpression extends AbstractUnaryExpression implements Serializable {

    /**
     * Power of norm.
     *
     */
    private final int p;

    /**
     * Scalar matrix corresponding p value.
     *
     */
    private final Matrix pMinusMatrix;

    /**
     * Constructor for variance operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @param p power of norm.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public NormExpression(int expressionID, Node argument1, Node result, int p) throws MatrixException {
        super(expressionID, argument1, result);
        this.p = p;
        this.pMinusMatrix = new DMatrix(p - 1);
    }

    /**
     * Calculates expression.
     *
     */
    public void calculateExpression() {
    }

    /**
     * Calculates expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int index) throws MatrixException {
        if (argument1.getMatrix(index) == null) throw new MatrixException("Arguments for NORM operation not defined");
        result.setMatrix(index, argument1.getMatrix(index).normAsMatrix(p));
    }

    /**
     * Calculates gradient of expression.
     *
     */
    public void calculateGradient() {
    }

    /**
     * Calculates gradient of expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int index) throws MatrixException {
        if (result.getGradient(index) == null) throw new MatrixException("Result gradient not defined.");
        // https://math.stackexchange.com/questions/1482494/derivative-of-the-l-p-norm/1482525
        argument1.updateGradient(index, result.getGradient(index).multiply(argument1.getMatrix(index).apply(UnaryFunctionType.ABS).divide(result.getMatrix(index)).applyBi(pMinusMatrix, BinaryFunctionType.POW).multiply(argument1.getMatrix(index).apply(UnaryFunctionType.SGN))), true);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        System.out.print("Expression " +getExpressionID() + ": ");
        System.out.println("NORM(" + p + ", " + argument1.getName() + ") = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        System.out.print("Expression " +getExpressionID() + ": ");
        System.out.println("NORM: d" + argument1.getName() + " = d" + result.getName() + " * ABS(" + argument1.getName() + ") / (" + result.getName() + "^" + (p - 1) + " * SGN(" + argument1.getName()   +"))");
    }

}
