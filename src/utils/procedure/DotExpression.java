/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import utils.matrix.MatrixException;

import java.io.Serializable;

/**
 * Class that describes expression for dot operation.
 *
 */
public class DotExpression extends AbstractBinaryExpression implements Serializable {

    /**
     * Constructor for dot operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public DotExpression(int expressionID, Node argument1, Node argument2, Node result) throws MatrixException {
        super(expressionID, argument1, argument2, result);
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
        if (argument1.getMatrix(index) == null || argument2.getMatrix(index) == null) throw new MatrixException("Arguments for DOT operation not defined");
        result.setMatrix(index, argument1.getMatrix(index).dot(argument2.getMatrix(index)));
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
        argument1.updateGradient(index, result.getGradient(index).dot(argument2.getMatrix(index).transpose()), true);
        argument2.updateGradient(index, argument1.getMatrix(index).transpose().dot(result.getGradient(index)), true);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        System.out.print("Expression " +getExpressionID() + ": ");
        System.out.println("DOT: " + argument1.getName() + " x " + argument2.getName() + " = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        System.out.print("Expression " +getExpressionID() + ": ");
        System.out.println("DOT: d" + argument1.getName() + " = d" + result.getName() + " x " + argument2.getName() + ".T");
        System.out.print("Expression " +getExpressionID() + ": ");
        System.out.println("DOT: d" + argument2.getName() + " = " + argument1.getName() + ".T x d" + result.getName());
    }

}
