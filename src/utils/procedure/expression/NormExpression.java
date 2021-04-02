/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.*;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Class that defines norm expression.
 *
 */
public class NormExpression extends AbstractUnaryExpression implements Serializable {

    /**
     * Name of operation.
     *
     */
    private static final String expressionName = "NORM";

    /**
     * Power of norm.
     *
     */
    private final int p;

    /**
     * Constructor for norm operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @param p power of norm.
     * @throws MatrixException throws exception if expression arguments are not defined or norm p value is not at least 2.
     */
    public NormExpression(int expressionID, Node argument1, Node result, int p) throws MatrixException {
        super(expressionName, expressionName, expressionID, argument1, result);
        if (p < 2) throw new MatrixException("Norm p value must be at least 2.");
        this.p = p;
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
        if (argument1.getMatrix(index) == null) throw new MatrixException(expressionName + "Arguments for operation not defined");
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
        if (result.getGradient(index) == null) throw new MatrixException(expressionName + ": Result gradient not defined.");
        // https://math.stackexchange.com/questions/1482494/derivative-of-the-l-p-norm/1482525
        argument1.updateGradient(index, result.getGradient(index).multiply(argument1.getMatrix(index).applyBi(result.getMatrix(index), (value, constant) -> Math.pow(Math.abs(value) / constant, p - 1) * Math.signum(value))), true);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        print();
        System.out.println(getName() + "(" + p + ", " + argument1.getName() + ") = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(true, " * (ABS(" + argument1.getName() + ")" + " / " + result.getName() + ")^" + (p - 1) + " * SGN("  + argument1.getName() + ")");
    }

}
