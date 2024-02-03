/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.operation.AveragePoolGradientMatrixOperation;
import utils.matrix.operation.AveragePoolMatrixOperation;
import utils.procedure.node.Node;

/**
 * Implements expression for average pooling operation.<br>
 *
 */
public class AveragePoolExpression extends AbstractUnaryExpression {

    /**
     * Reference to average pool matrix operation.
     *
     */
    private final AveragePoolMatrixOperation averagePoolMatrixOperation;

    /**
     * Reference to average pool gradient matrix operation.
     *
     */
    private final AveragePoolGradientMatrixOperation averagePoolGradientMatrixOperation;

    /**
     * Constructor for average pool expression.
     *
     * @param expressionID     unique ID for expression.
     * @param argument1        first argument.
     * @param result           result of expression.
     * @param dilation         dilation of pooling operation.
     * @param stride           stride of pooling operation.
     * @param filterRowSize    filter row size for operation.
     * @param filterColumnSize filter column size for operation.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public AveragePoolExpression(int expressionID, Node argument1, Node result, int dilation, int stride, int filterRowSize, int filterColumnSize) throws MatrixException {
        super("AVERAGE_POOL", expressionID, argument1, result);

        averagePoolMatrixOperation = new AveragePoolMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), filterRowSize, filterColumnSize, dilation, stride);
        averagePoolGradientMatrixOperation = new AveragePoolGradientMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), filterRowSize, filterColumnSize, dilation, stride);
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
        return averagePoolMatrixOperation.apply(argument1Matrix);
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
        return averagePoolGradientMatrixOperation.apply(resultGradient);
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
        return getExpressionName() + "_GRADIENT(d" + getResult().getName() + ")";
    }

}
