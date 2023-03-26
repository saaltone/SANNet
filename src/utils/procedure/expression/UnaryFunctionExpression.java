/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.*;
import utils.matrix.operation.UnaryMatrixOperation;
import utils.procedure.node.Node;

/**
 * Implements expression for unary function.<br>
 *
 */
public class UnaryFunctionExpression extends AbstractUnaryExpression {

    /**
     * Unary function type.
     *
     */
    private final UnaryFunctionType unaryFunctionType;

    /**
     * UnaryFunction used.
     *
     */
    @SuppressWarnings("FieldCanBeLocal")
    private final UnaryFunction unaryFunction;

    /**
     * Unary matrix operation.
     *
     */
    private final UnaryMatrixOperation unaryMatrixOperation;

    /**
     * Constructor for unary function.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result.
     * @param unaryFunction UnaryFunction.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public UnaryFunctionExpression(int expressionID, Node argument1, Node result, UnaryFunction unaryFunction) throws MatrixException {
        super("UNARY_FUNCTION", String.valueOf(unaryFunction.getType()), expressionID, argument1, result);

        this.unaryFunctionType = unaryFunction.getType();
        this.unaryFunction = unaryFunction;

        unaryMatrixOperation = new UnaryMatrixOperation(argument1.getRows(), argument1.getColumns(), argument1.getDepth(), unaryFunction);
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
        checkArgument(argument1, sampleIndex);
        result.setMatrix(sampleIndex, unaryMatrixOperation.applyFunction(argument1.getMatrix(sampleIndex)));
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
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, unaryMatrixOperation.applyGradient(result.getMatrix(sampleIndex), result.getGradient(sampleIndex)), false);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        print();
        System.out.println(getExpressionName() + ": " + unaryFunctionType + "(" + argument1.getName() + ") = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(true, " * " + unaryFunctionType + "_GRADIENT(" + result.getName() + ")");
    }

}
