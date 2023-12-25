/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.operation.DotMatrixOperation;
import utils.procedure.node.Node;

/**
 * Implements expression for dot operation.<br>
 *
 */
public class DotExpression extends AbstractBinaryExpression {

    /**
     * Reference to dot matrix operation.
     *
     */
    private final DotMatrixOperation dotMatrixOperation;

    /**
     * Reference to dot gradient 1 matrix operation.
     *
     */
    private final DotMatrixOperation dotGradient1MatrixOperation;

    /**
     * Reference to dot gradient 2 matrix operation.
     *
     */
    private final DotMatrixOperation dotGradient2MatrixOperation;

    /**
     * Constructor for dot operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public DotExpression(int expressionID, Node argument1, Node argument2, Node result) throws MatrixException {
        super("DOT", expressionID, argument1, argument2, result);

        dotMatrixOperation = new DotMatrixOperation(argument1.getRows(), argument2.getRows(), argument2.getColumns(), argument1.getDepth());
        dotGradient1MatrixOperation = new DotMatrixOperation(result.getRows(), argument2.getColumns(), argument2.getRows(), argument2.getDepth());
        dotGradient2MatrixOperation = new DotMatrixOperation(argument1.getColumns(), result.getRows(), result.getColumns(), argument1.getDepth());
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
        return dotMatrixOperation.apply(argument1Matrix, argument2Matrix);
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
        return dotGradient1MatrixOperation.apply(resultGradient, argument2Matrix.transpose());
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
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected Matrix calculateArgument2Gradient(int sampleIndex, Matrix resultGradient, Matrix argument1Matrix, Matrix argument2Matrix, Matrix resultMatrix) throws MatrixException, DynamicParamException {
        return dotGradient2MatrixOperation.apply(argument1Matrix.transpose(), resultGradient);
    }

    /**
     * Returns expression operation signature.
     *
     * @return expression operation signature.
     */
    protected String getExpressionOperationSignature() {
        return getArgument1().getName() + " x " + getArgument2().getName();
    }

    /**
     * Returns gradient 1 operation signature.
     *
     * @return gradient 1 operation signature.
     */
    protected String getGradientOperation1Signature() {
        return "d" + getResult().getName() + " x " + getArgument2().getName() + ".T";
    }

    /**
     * Returns gradient 2 operation signature.
     *
     * @return gradient 2 operation signature.
     */
    protected String getGradientOperation2Signature() {
        return getArgument1().getName() + ".T" + " x d" + getResult().getName();
    }

}
