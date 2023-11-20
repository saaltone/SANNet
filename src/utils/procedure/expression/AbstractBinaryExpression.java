/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.procedure.node.Node;

/**
 * Implements abstract binary expression.<br>
 *
 */
public abstract class AbstractBinaryExpression extends AbstractUnaryExpression {

    /**
     * Node for second argument.
     *
     */
    protected final Node argument2;

    /**
     * Constructor for abstract binary expression.
     *
     * @param name               name of expression.
     * @param expressionID       expression ID
     * @param argument1          first argument.
     * @param argument2          second argument.
     * @param result             result of node.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public AbstractBinaryExpression(String name, int expressionID, Node argument1, Node argument2, Node result) throws MatrixException {
        super(name, expressionID, argument1, result);
        if (argument2 == null) throw new MatrixException("Second argument not defined.");
        this.argument2 = argument2;
    }

    /**
     * Returns second argument of expression.
     *
     * @return second argument of expression.
     */
    public Node getArgument2() {
        return argument2;
    }

    /**
     * Calculates expression.
     *
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int sampleIndex) throws MatrixException {
        if (executeAsSingleStep()) return;
        checkArguments(getArgument1(), getArgument2(), sampleIndex);
        calculateExpressionResult(sampleIndex, getArgument1().getMatrix(sampleIndex), getArgument2().getMatrix(sampleIndex));
    }

    /**
     * Calculates gradient of expression.
     *
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int sampleIndex) throws MatrixException {
        if (executeAsSingleStep()) return;
        super.calculateGradient(sampleIndex);
        cumulateArgument2Gradient(sampleIndex);
    }

    /**
     * Cumulates argument 2 gradient.
     *
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void cumulateArgument2Gradient(int sampleIndex) throws MatrixException {
        if (!argument2.isStopGradient()) argument2.cumulateGradient(sampleIndex, calculateArgument2Gradient(sampleIndex, getResult().getGradient(sampleIndex), getArgument1().getMatrix(sampleIndex), getArgument2().getMatrix(sampleIndex), getResult().getMatrix(sampleIndex)));
    }

    /**
     * Calculates argument 2 gradient matrix.
     *
     * @param sampleIndex     sample index.
     * @param resultGradient  result gradient.
     * @param argument1Matrix argument 1 matrix.
     * @param argument2Matrix argument 2 matrix.
     * @param resultMatrix    result matrix.
     * @return argument1 gradient matrix.
     * @throws MatrixException throws exception if calculation fails.
     */
    protected abstract Matrix calculateArgument2Gradient(int sampleIndex, Matrix resultGradient, Matrix argument1Matrix, Matrix argument2Matrix, Matrix resultMatrix) throws MatrixException;

    /**
     * Check is argument matrices are defined for specific sample index.
     *
     * @param argument1 argument 1
     * @param argument2 argument 2
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if one of both arguments are not defined.
     */
    protected void checkArguments(Node argument1, Node argument2, int sampleIndex) throws MatrixException {
        if (argument1.getMatrix(sampleIndex) == null && argument2.getMatrix(sampleIndex) == null) throw new MatrixException(getExpressionName() + ": Arguments 1 and 2 for operation are not defined for sample index " + sampleIndex);
        checkArgument(argument1, sampleIndex);
        if (argument2.getMatrix(sampleIndex) == null) throw new MatrixException(getExpressionName() + ": Argument 2 for operation is not defined for sample index " + sampleIndex);
    }

    /**
     * Prints gradient.
     *
     */
    protected void printGradient() {
        super.printGradient();
        if (getGradientOperation2Signature() != null) {
            System.out.println(getGradientOperationSignature(getArgument2(), getGradientOperation2Signature()));
        }
    }

    /**
     * Returns gradient 2 operation signature.
     *
     * @return gradient 2 operation signature.
     */
    protected abstract String getGradientOperation2Signature();

}
