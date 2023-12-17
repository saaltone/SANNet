/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.procedure.node.Node;

/**
 * Implements abstract unary expression.<br>
 *
 */
public abstract class AbstractUnaryExpression extends AbstractExpression {

    /**
     * Node for first argument.
     *
     */
    protected final Node argument1;

    /**
     * Node for result.
     *
     */
    protected final Node result;

    /**
     * Constructor for abstract unary expression.
     *
     * @param name               name of expression.
     * @param expressionID       expression ID
     * @param argument1          first argument of expression.
     * @param result             result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public AbstractUnaryExpression(String name, int expressionID, Node argument1, Node result) throws MatrixException {
        super(name, expressionID, argument1);
        this.argument1 = argument1;
        this.result = result;
    }

    /**
     * Returns first argument of expression.
     *
     * @return first argument of expression.
     */
    public Node getArgument1() {
        return argument1;
    }

    /**
     * Checks if argument matrix is defined for specific sample index.
     *
     * @param argument1 argument1
     * @param sampleIndex sample index.
     * @throws MatrixException throws exception if argument is not defined.
     */
    protected void checkArgument(Node argument1, int sampleIndex) throws MatrixException {
        if (argument1.getMatrix(sampleIndex) == null) throw new MatrixException(this + ": " + getExpressionName() + ": Argument 1 for operation is not defined for sample index " + sampleIndex);
    }

    /**
     * Returns second argument of expression.
     *
     * @return returns null unless overloaded by abstract binary expression class.
     */
    public Node getArgument2() {
        return null;
    }

    /**
     * Returns result of expression.
     *
     * @return result of expression.
     */
    public Node getResult() {
        return result;
    }

    /**
     * Calculates expression.
     *
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateExpression() throws MatrixException, DynamicParamException {
        if (!executeAsSingleStep()) return;
        if (argument1.getMatrices() == null) throw new MatrixException(getExpressionName() + ": Argument 1 for operation not defined");
        result.setMatrix(calculateResult());
    }

    /**
     * Calculates result matrix.
     *
     * @return result matrix.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected abstract Matrix calculateResult() throws MatrixException, DynamicParamException;

    /**
     * Calculates expression.
     *
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateExpression(int sampleIndex) throws MatrixException, DynamicParamException {
        if (executeAsSingleStep()) return;
        checkArgument(getArgument1(), sampleIndex);
        calculateExpressionResult(sampleIndex, getArgument1().getMatrix(sampleIndex), null);
    }

    /**
     * Calculates expression result.
     *
     * @param sampleIndex sample index
     * @param argument1Matrix argument1 matrix for a sample index.
     * @param argument2Matrix argument2 matrix for a sample index.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void calculateExpressionResult(int sampleIndex, Matrix argument1Matrix, Matrix argument2Matrix) throws MatrixException, DynamicParamException {
        result.setMatrix(sampleIndex, calculateResult(sampleIndex, argument1Matrix, argument2Matrix));
    }

    /**
     * Calculates result matrix.
     *
     * @param sampleIndex sample index
     * @param argument1Matrix argument1 matrix for a sample index.
     * @param argument2Matrix argument2 matrix for a sample index.
     * @return result matrix.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected abstract Matrix calculateResult(int sampleIndex, Matrix argument1Matrix, Matrix argument2Matrix) throws MatrixException, DynamicParamException;

    /**
     * Calculates gradient of expression.
     *
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateGradient() throws MatrixException {
        if (!executeAsSingleStep()) return;
        if (result.getGradient() == null) throw new MatrixException(getExpressionName() + ": Result gradient not defined");
        calculateArgument1Gradient();
    }

    /**
     * Calculates argument 1 gradient matrix.
     *
     * @throws MatrixException throws exception if calculation fails.
     */
    protected abstract void calculateArgument1Gradient() throws MatrixException;

    /**
     * Calculates gradient of expression.
     *
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int sampleIndex) throws MatrixException {
        if (executeAsSingleStep()) return;
        checkResultGradient(result, sampleIndex);
        cumulateArgument1Gradient(sampleIndex);
    }

    /**
     * Cumulates argument 1 gradient.
     *
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void cumulateArgument1Gradient(int sampleIndex) throws MatrixException {
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, calculateArgument1Gradient(sampleIndex, getResult().getGradient(sampleIndex), getArgument1().getMatrix(sampleIndex), getArgument2() != null ? getArgument2().getMatrix(sampleIndex) : null, getResult().getMatrix(sampleIndex)));
    }

    /**
     * Calculates argument1 gradient matrix.
     *
     * @param sampleIndex     sample index.
     * @param resultGradient  result gradient.
     * @param argument1Matrix argument 1 matrix.
     * @param argument2Matrix argument 2 matrix.
     * @param resultMatrix    result matrix.
     * @return argument1 gradient matrix.
     * @throws MatrixException throws exception if calculation fails.
     */
    protected abstract Matrix calculateArgument1Gradient(int sampleIndex, Matrix resultGradient, Matrix argument1Matrix, Matrix argument2Matrix, Matrix resultMatrix) throws MatrixException;

    /**
     * Prints gradient.
     *
     */
    protected void printGradient() {
        if (getGradientOperation1Signature() != null) {
            System.out.println(getGradientOperationSignature(getArgument1(), getGradientOperation1Signature()));
        }
    }

    /**
     * Return gradient operation signature.
     *
     * @param argument node argument.
     * @param gradientOperationSignature gradient operation signature.
     * @return gradient operation signature.
     */
    protected String getGradientOperationSignature(Node argument, String gradientOperationSignature) {
        return "Gradient" + getExpressionID() + ": " + getExpressionName() + ": " + (!argument.isMultiIndex() ? "sum(" + gradientOperationSignature + ")" : gradientOperationSignature) + " = d" + argument.getName() + (argument.isStopGradient() ? " [ stop gradient ]" : "");
    }

    /**
     * Returns gradient 1 operation signature.
     *
     * @return gradient 1 operation signature.
     */
    protected abstract String getGradientOperation1Signature();

}
