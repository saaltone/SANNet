/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.Matrix;
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
        super("GRADIENT_CLIPPING", expressionID, argument1, result);

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
     * Calculates result matrix.
     *
     * @return result matrix.
     */
    protected Matrix calculateResult() {
        return null;
    }

    /**
     * Calculates result matrix.
     *
     * @param sampleIndex sample index
     * @param argument1Matrix argument1 matrix for a sample index.
     * @param argument2Matrix argument2 matrix for a sample index.
     * @return result matrix.
     */
    protected Matrix calculateResult(int sampleIndex, Matrix argument1Matrix, Matrix argument2Matrix) {
        return argument1Matrix;
    }

    /**
     * Calculates argument 1 gradient matrix.
     */
    protected void calculateArgument1Gradient() {
    }

    /**
     * Calculates argument 1 gradient matrix.
     *
     * @param sampleIndex     sample index.
     * @param resultGradient  result gradient.
     * @param argument1Matrix argument 1 matrix.
     * @param argument2Matrix argument 2 matrix.
     * @param resultMatrix    result matrix.
     * @return argument1 gradient matrix.
     * @throws MatrixException throws exception if calculation fails.
     */
    protected Matrix calculateArgument1Gradient(int sampleIndex, Matrix resultGradient, Matrix argument1Matrix, Matrix argument2Matrix, Matrix resultMatrix) throws MatrixException {
        return gradientClippingMatrixOperation.apply(resultGradient, false);
    }

    /**
     * Returns expression operation signature.
     *
     * @return expression operation signature.
     */
    protected String getExpressionOperationSignature() {
        return getArgument1().getName();
    }

    /**
     * Returns gradient 1 operation signature.
     *
     * @return gradient 1 operation signature.
     */
    protected String getGradientOperation1Signature() {
        return getExpressionName() + "(" + "d" + getResult().getName() + ", " + threshold + ")";
    }

}
