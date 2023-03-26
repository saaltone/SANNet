/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.matrix.operation.JoinMatrixOperation;
import utils.procedure.node.Node;

/**
 * Implements expression for join function.<br>
 *
 */
public class JoinExpression extends AbstractBinaryExpression {

    /**
     * Reference to join matrix operation.
     *
     */
    private final JoinMatrixOperation joinMatrixOperation;

    /**
     * Constructor for join function.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result.
     * @param joinedVertically if true joined vertically otherwise horizontally
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public JoinExpression(int expressionID, Node argument1, Node argument2, Node result, boolean joinedVertically) throws MatrixException {
        super("JOIN", joinedVertically ? "VERTICALLY" : "HORIZONTALLY", expressionID, argument1, argument2, result);

        joinMatrixOperation = new JoinMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), joinedVertically);
    }

    /**
     * Returns true is expression is executed as single step otherwise false.
     *
     * @return true is expression is executed as single step otherwise false.
     */
    protected boolean executeAsSingleStep() {
        return false;
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
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int sampleIndex) throws MatrixException {
        checkArguments(argument1, argument2, sampleIndex);
        result.setMatrix(sampleIndex, joinMatrixOperation.apply(argument1.getMatrix(sampleIndex), argument2.getMatrix(sampleIndex)));
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
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int sampleIndex) throws MatrixException {
        checkResultGradient(result, sampleIndex);
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, joinMatrixOperation.applyGradient(result.getGradient(sampleIndex), true), false);
        if (!argument2.isStopGradient()) argument2.cumulateGradient(sampleIndex, joinMatrixOperation.applyGradient(result.getGradient(sampleIndex), false), false);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        print();
        System.out.println(getExpressionName() + ": " + "JOIN(" + argument1.getName() + " & " + argument2.getName() + ") = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(true, "");
        printArgument2Gradient(false, false, getResultGradientName());
    }

}
