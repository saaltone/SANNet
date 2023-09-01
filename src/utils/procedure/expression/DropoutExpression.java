/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.matrix.operation.DropoutMatrixOperation;
import utils.procedure.node.Node;

/**
 * Implements dropout expression.<br>
 *
 */
public class DropoutExpression extends AbstractUnaryExpression {

    /**
     * Reference to dropout matrix operation.
     *
     */
    private final DropoutMatrixOperation dropoutMatrixOperation;

    /**
     * Probability of dropout.
     *
     */
    private final double probability;

    /**
     * If true dropout is monte carlo dropout otherwise normal dropout.
     *
     */
    private final boolean monte_carlo;

    /**
     * Constructor for dropout operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @param probability probability
     * @param monte_carlo if true is monte carlo dropout otherwise normal dropout.
     * @throws MatrixException throws exception if expression arguments are not defined or norm p value is not at least 2.
     */
    public DropoutExpression(int expressionID, Node argument1, Node result, double probability, boolean monte_carlo) throws MatrixException {
        super("DROPOUT", "DROPOUT", expressionID, argument1, result);

        if (probability < 0 || probability > 1) throw new MatrixException("Probability must be between 0 and 1.");
        this.probability = probability;
        this.monte_carlo = monte_carlo;

        dropoutMatrixOperation = new DropoutMatrixOperation(argument1.getRows(), argument1.getColumns(), argument1.getDepth(), probability);
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
        if (isActive() || monte_carlo) result.setMatrix(sampleIndex, dropoutMatrixOperation.apply(argument1.getMatrix(sampleIndex), false));
        else result.setMatrix(sampleIndex, argument1.getMatrix(sampleIndex));
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
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, result.getMatrix(sampleIndex), false);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        print();
        System.out.println(getExpressionName() + "(" + probability + ", " + argument1.getName() + ") = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(false, "d" + result.getName());
    }

}
