/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.matrix.operation.GradientClippingMatrixOperation;
import utils.procedure.node.Node;

/**
 * Implements gradient clipping operation
 *
 */
public class GradientClippingExpression extends AbstractUnaryExpression {

    /**
     * Reference to gradient clipping matrix operation.
     *
     */
    private final GradientClippingMatrixOperation gradientClippingMatrixOperation;


    /**
     * Threshold.
     *
     */
    private final double threshold;

    /**
     * Constructor for gradient clipping operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @param threshold threshold.
     * @throws MatrixException throws exception if expression arguments are not defined or norm p value is not at least 2.
     */
    public GradientClippingExpression(int expressionID, Node argument1, Node result, double threshold) throws MatrixException {
        super("GRADIENT_CLIPPING", "GRADIENT_CLIPPING", expressionID, argument1, result);

        this.threshold = threshold;

        gradientClippingMatrixOperation = new GradientClippingMatrixOperation(argument1.getRows(), argument1.getColumns(), argument1.getDepth(), threshold);
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
        result.setMatrix(sampleIndex, argument1.getMatrix(sampleIndex));
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
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, gradientClippingMatrixOperation.apply(result.getMatrix(sampleIndex), false), false);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        print();
        System.out.println(argument1.getName() + " = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(false, getExpressionName() + "(" + threshold + ", d" + result.getName() + ")");
    }

}
