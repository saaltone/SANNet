/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Class that describes expression for dot operation.<br>
 *
 */
public class DotExpression extends AbstractBinaryExpression implements Serializable {

    /**
     * Name of expression.
     *
     */
    private static final String expressionName = "DOT";

    /**
     * Operation signature.
     *
     */
    private static final String operationSignature = "x";

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
        super(expressionName, operationSignature, expressionID, argument1, argument2, result);
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
        if (argument1.getMatrix(index) == null || argument2.getMatrix(index) == null) throw new MatrixException(expressionName + "Arguments for operation not defined");
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
        if (result.getGradient(index) == null) throw new MatrixException(expressionName + ": Result gradient not defined.");
        argument1.cumulateGradient(index, result.getGradient(index).dot(argument2.getMatrix(index).transpose()), false);
        argument2.cumulateGradient(index, argument1.getMatrix(index).transpose().dot(result.getGradient(index)), false);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        printBasicBinaryExpression();
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(true, " " + operationSignature + " " + argument2.getName() + ".T");
        printArgument2Gradient(false, false, argument1.getName() + ".T" + " " + operationSignature + " " + getResultGradientName());
    }

}
