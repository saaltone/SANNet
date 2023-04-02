/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.matrix.operation.FlattenMatrixOperation;
import utils.matrix.operation.UnflattenMatrixOperation;
import utils.procedure.node.Node;

/**
 * Implements expression for matrix flatten operation.
 *
 */
public class FlattenExpression extends AbstractUnaryExpression {

    /**
     * Reference to flatten matrix operation.
     *
     */
    private final FlattenMatrixOperation flattenMatrixOperation;

    /**
     * Reference to unflatten matrix operation.
     *
     */
    private final UnflattenMatrixOperation unflattenMatrixOperation;

    /**
     * Constructor for flatten matrix operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public FlattenExpression(int expressionID, Node argument1, Node result) throws MatrixException {
        super("FLATTEN", "FLATTEN", expressionID, argument1, result);

        flattenMatrixOperation = new FlattenMatrixOperation(argument1.getRows(), argument1.getColumns(), argument1.getDepth());
        unflattenMatrixOperation = new UnflattenMatrixOperation(argument1.getRows(), argument1.getColumns(), argument1.getDepth());
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
     * Resets expression.
     *
     */
    public void applyReset() {
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
        result.setMatrix(sampleIndex, flattenMatrixOperation.apply(argument1.getMatrix(sampleIndex)));
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
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, unflattenMatrixOperation.apply(result.getGradient(sampleIndex)), false);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        print();
        System.out.println(getExpressionName() + "(" + argument1.getName() + ") = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(false, "UN" + getExpressionName() + "(" + getResultGradientName() + ")");
    }

}
