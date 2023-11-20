/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.Matrix;
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
        super("DROPOUT", expressionID, argument1, result);

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
     * @throws MatrixException throws exception if calculation fails.
     */
    protected Matrix calculateResult(int sampleIndex, Matrix argument1Matrix, Matrix argument2Matrix) throws MatrixException {
        return isActive() || monte_carlo ? dropoutMatrixOperation.apply(argument1Matrix, false) : argument1Matrix;
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
     */
    protected Matrix calculateArgument1Gradient(int sampleIndex, Matrix resultGradient, Matrix argument1Matrix, Matrix argument2Matrix, Matrix resultMatrix) {
        return resultGradient;
    }

    /**
     * Returns expression operation signature.
     *
     * @return expression operation signature.
     */
    protected String getExpressionOperationSignature() {
        return getExpressionName() + "(" + probability + ", "  + getArgument1().getName() + ")";
    }

    /**
     * Returns gradient 1 operation signature.
     *
     * @return gradient 1 operation signature.
     */
    protected String getGradientOperation1Signature() {
        return "d" + getResult().getName();
    }

}
