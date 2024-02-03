/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.AbstractMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.procedure.node.Node;

/**
 * Implements expression for mean function.<br>
 *
 */
public class MeanExpression extends AbstractUnaryExpression {

    /**
     * If value is one applies operation over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     *
     */
    private final int direction;

    /**
     * True if calculation is done as single step otherwise false.
     *
     */
    private final boolean executeAsSingleStep;

    /**
     * Constructor for mean operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @param executeAsSingleStep true if calculation is done per index otherwise over all indices.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public MeanExpression(int expressionID, Node argument1, Node result, boolean executeAsSingleStep, int direction) throws MatrixException {
        super("MEAN", expressionID, argument1, result);

        this.executeAsSingleStep = executeAsSingleStep;
        this.direction = direction;
    }

    /**
     * Returns true is expression is executed as single step otherwise false.
     *
     * @return true is expression is executed as single step otherwise false.
     */
    protected boolean executeAsSingleStep() {
        return executeAsSingleStep;
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
     * @throws MatrixException throws exception if calculation fails.
     */
    protected Matrix calculateResult() throws MatrixException {
        return AbstractMatrix.mean(argument1.getMatrices());
    }

    /**
     * Calculates result matrix.
     *
     * @param sampleIndex sample index
     * @param argument1Matrix argument1 matrix for a sample index.
     * @param argument2Matrix argument2 matrix for a sample index.
     * @return result matrix.
     * @throws MatrixException throws exception if calculation fails.
     */
    protected Matrix calculateResult(int sampleIndex, Matrix argument1Matrix, Matrix argument2Matrix) throws MatrixException {
        return argument1Matrix.meanAsMatrix(direction);
    }

    /**
     * Calculates argument 1 gradient matrix.
     *
     * @throws MatrixException throws exception if calculation fails.
     */
    protected void calculateArgument1Gradient() throws MatrixException {
        Matrix meanGradient = result.getGradient().multiply(1 / (double)argument1.size());
        for (Integer index : argument1.keySet()) argument1.cumulateGradient(index, meanGradient);
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
        return resultGradient.multiply(1 / (double)argument1Matrix.size());
    }

    /**
     * Returns expression operation signature.
     *
     * @return expression operation signature.
     */
    protected String getExpressionOperationSignature() {
        return getExpressionName() + "(" + getArgument1().getName() + ")";
    }

    /**
     * Returns gradient 1 operation signature.
     *
     * @return gradient 1 operation signature.
     */
    protected String getGradientOperation1Signature() {
        return "d" + getResult().getName() + " / SIZE(" + getArgument1().getName() + ")";
    }

}
