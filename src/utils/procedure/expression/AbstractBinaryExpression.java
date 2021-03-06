/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.DynamicParamException;
import utils.matrix.MatrixException;
import utils.procedure.node.Node;

/**
 * Class that defines AbstractBinaryExpression.<br>
 *
 */
public abstract class AbstractBinaryExpression extends AbstractExpression {

    /**
     * Node for second argument.
     *
     */
    protected final Node argument2;

    /**
     * Constructor for AbstractBinaryExpression.
     *
     * @param name name of expression.
     * @param operationSignature operation signature
     * @param expressionID expression ID
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of node.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public AbstractBinaryExpression(String name, String operationSignature, int expressionID, Node argument1, Node argument2, Node result) throws MatrixException {
        super(name, operationSignature, expressionID, argument1, result);
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
     * Updates expression forward direction dependency.
     *
     * @param index index
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching or node is of type multi-index.
     */
    protected void updateExpressionDependency(int index) throws MatrixException {
        super.updateExpressionDependency(index);
        argument2.updateMatrixDependency(index);
    }

    /**
     * Make forward normalize callback to all entries of node.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardNormalize() throws MatrixException, DynamicParamException {
        super.forwardNormalize();
        argument2.forwardNormalize();
    }

    /**
     * Make forward normalize callback to specific entry (sample)
     *
     * @param sampleIndex sample index of specific entry.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardNormalize(int sampleIndex) throws MatrixException, DynamicParamException {
        super.forwardNormalize(sampleIndex);
        argument2.forwardNormalize(sampleIndex);
    }

    /**
     * Make backward normalize callback to all entries of node.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void backwardNormalize() throws MatrixException, DynamicParamException {
        super.backwardNormalize();
        argument2.backwardNormalize();
    }

    /**
     * Make backward normalize callback to specific entry (sample)
     *
     * @param sampleIndex sample index of specific entry.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void backwardNormalize(int sampleIndex) throws MatrixException, DynamicParamException {
        super.backwardNormalize(sampleIndex);
        argument2.backwardNormalize(sampleIndex);
    }

    /**
     * Executes forward regularization step.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardRegularize() throws MatrixException {
        super.forwardRegularize();
        argument2.forwardRegularize();
    }

    /**
     * Cumulates error from regularization.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return updated error value.
     */
    public double cumulateRegularizationError() throws MatrixException, DynamicParamException {
        return super.cumulateRegularizationError() + argument2.cumulateRegularizationError();
    }

    /**
     * Executes backward regularization.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardRegularize() throws MatrixException {
        super.backwardRegularize();
        argument2.backwardRegularize();
    }

    /**
     * Print basic binary expression.
     *
     */
    protected void printBasicBinaryExpression() {
        print();
        System.out.println(getExpressionName() + ": " + argument1.getName() + " " + getOperationSignature() + " " + argument2.getName() + " = " + result.getName());
    }

    /**
     * Print basic binary expression.
     *
     */
    protected void printSpecificBinaryExpression() {
        print();
        System.out.println(getExpressionName() + ": " + getOperationSignature() + "(" + argument1.getName() + ", " + " " + argument2.getName() + ") = " + result.getName());
    }

    /**
     * Prints gradient for argument2
     *
     * @param withResultPrefix true if result prefix is added.
     * @param negateResult true if result is negated.
     * @param suffix suffix part for gradient expression.
     */
    protected void printArgument2Gradient(boolean withResultPrefix, boolean negateResult, String suffix) {
        print();
        System.out.println((withResultPrefix ? getNodeGradientWithResultPrefixName(argument2, negateResult) : getNodeGradientPrefixName(argument2, negateResult)) + (suffix != null ? "" + suffix : "") + getNodeSumPostfix(argument2));
    }

}
