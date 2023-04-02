/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
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
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public MeanExpression(int expressionID, Node argument1, Node result, boolean executeAsSingleStep) throws MatrixException {
        super("MEAN", "MEAN", expressionID, argument1, result);

        this.executeAsSingleStep = executeAsSingleStep;
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
     * Calculates expression.
     *
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression() throws MatrixException {
        if (!executeAsSingleStep()) return;
        if (argument1.getMatrices() == null) throw new MatrixException(getExpressionName() + ": Argument 1 for operation not defined");
        result.setMatrix(AbstractMatrix.mean(argument1.getMatrices()));
    }

    /**
     * Calculates expression.
     *
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int sampleIndex) throws MatrixException {
        if (executeAsSingleStep()) return;
        checkArgument(argument1, sampleIndex);
        result.setMatrix(sampleIndex, argument1.getMatrix(sampleIndex).meanAsMatrix());
    }

    /**
     * Calculates gradient of expression.
     *
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateGradient() throws MatrixException {
        if (!executeAsSingleStep()) return;
        if (result.getGradient() == null) throw new MatrixException(getExpressionName() + ": Result gradient not defined");
        Matrix meanGradient = result.getGradient().multiply(1 / (double)argument1.size());
        for (Integer index : argument1.keySet()) argument1.cumulateGradient(index, meanGradient, false);
    }

    /**
     * Calculates gradient of expression.
     *
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int sampleIndex) throws MatrixException {
        if (executeAsSingleStep()) return;
        checkResultGradient(result, sampleIndex);
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, result.getGradient(sampleIndex).multiply(1 / (double)argument1.getMatrix(sampleIndex).size()), false);
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
        printArgument1Gradient(true, " / SIZE(" + argument1.getName() + ")");
    }

}
