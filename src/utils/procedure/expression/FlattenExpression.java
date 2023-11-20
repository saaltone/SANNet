/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.Matrix;
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
        super("FLATTEN", expressionID, argument1, result);

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
        return flattenMatrixOperation.apply(argument1Matrix);
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
        return unflattenMatrixOperation.apply(resultGradient);
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
        return "UN" + getExpressionName() + "(d" + getResult().getName() + ")";
    }

}
