/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.matrix.operation.UnaryMatrixOperation;
import utils.procedure.node.Node;

/**
 * Implements expression for unary function.<br>
 *
 */
public class UnaryFunctionExpression extends AbstractUnaryExpression {

    /**
     * Unary function type.
     *
     */
    private final UnaryFunctionType unaryFunctionType;

    /**
     * UnaryFunction used.
     *
     */
    @SuppressWarnings("FieldCanBeLocal")
    private final UnaryFunction unaryFunction;

    /**
     * Unary matrix operation.
     *
     */
    private final UnaryMatrixOperation unaryMatrixOperation;

    /**
     * Constructor for unary function.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result.
     * @param unaryFunction UnaryFunction.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public UnaryFunctionExpression(int expressionID, Node argument1, Node result, UnaryFunction unaryFunction) throws MatrixException {
        super("UNARY_FUNCTION", expressionID, argument1, result);

        this.unaryFunctionType = unaryFunction.getType();
        this.unaryFunction = unaryFunction;

        unaryMatrixOperation = new UnaryMatrixOperation(argument1.getRows(), argument1.getColumns(), argument1.getDepth(), unaryFunction);
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
        return unaryMatrixOperation.applyFunction(argument1Matrix);
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
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected Matrix calculateArgument1Gradient(int sampleIndex, Matrix resultGradient, Matrix argument1Matrix, Matrix argument2Matrix, Matrix resultMatrix) throws MatrixException, DynamicParamException {
        return unaryMatrixOperation.applyGradient(resultMatrix, resultGradient);
    }

    /**
     * Returns expression operation signature.
     *
     * @return expression operation signature.
     */
    protected String getExpressionOperationSignature() {
        return unaryFunctionType + "(" + getArgument1().getName() + ")";
    }

    /**
     * Returns gradient 1 operation signature.
     *
     * @return gradient 1 operation signature.
     */
    protected String getGradientOperation1Signature() {
        return "d" + getResult().getName() + " * " + unaryFunctionType + "_GRADIENT(" + getResult().getName() + ")";
    }

}
