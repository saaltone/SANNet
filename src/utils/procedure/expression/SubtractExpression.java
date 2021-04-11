/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Class that describes expression for subtraction operation.<br>
 *
 */
public class SubtractExpression extends AbstractBinaryExpression implements Serializable {

    /**
     * Name of expression.
     *
     */
    private static final String expressionName = "SUBTRACT";

    /**
     * Operation signature.
     *
     */
    private static final String operationSignature = "-";

    /**
     * Constructor for subtraction operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public SubtractExpression(int expressionID, Node argument1, Node argument2, Node result) throws MatrixException {
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
        result.setMatrix(index, argument1.getMatrix(index).subtract(argument2.getMatrix(index)));
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
        argument1.cumulateGradient(index, result.getGradient(index), false);
        argument2.cumulateGradient(index, result.getGradient(index), true);
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
        printArgument1Gradient(true, null);
        printArgument2Gradient(true, true, null);
    }

}
