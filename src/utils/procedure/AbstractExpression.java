/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.procedure;

import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.Set;

/**
 * Abstract lass that describes single computable expression including gradient expression.<br>
 * Assumes underlying class that implemented specific expression.<br>
 */
public abstract class AbstractExpression implements Serializable {

    private static final long serialVersionUID = -3692842009210981254L;

    /**
     * Name of expression;
     *
     */
    private final String name;

    /**
     * Operation signature.
     *
     */
    private final String operationSignature;

    /**
     * Unique ID of expression.
     *
     */
    private final int expressionID;

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
     * Next expression for expression calculation.
     *
     */
    private AbstractExpression nextExpression;

    /**
     * Previous expression for gradient calculation.
     *
     */
    private AbstractExpression previousExpression;

    /**
     * Constructor for abstract expression.
     *
     * @param name name of expression.
     * @param operationSignature operation signature of expression.
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public AbstractExpression(String name, String operationSignature, int expressionID, Node argument1, Node result) throws MatrixException {
        if (argument1 == null) throw new MatrixException("First argument not defined.");
        this.name = name;
        this.operationSignature = operationSignature;
        this.expressionID = expressionID;
        this.argument1 = argument1;
        this.result = result;
    }

    public String getName() {
        return name;
    }

    /**
     * Returns signature of operation.
     *
     * @return signature of operation.
     */
    public String getOperationSignature() {
        return operationSignature;
    }

    /**
     * Returns expression ID
     *
     * @return expression ID
     */
    public int getExpressionID() {
        return expressionID;
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
     * Sets next expression for expression calculation chain.
     *
     * @param nextExpression next expression.
     */
    public void setNextExpression(AbstractExpression nextExpression) {
        this.nextExpression = nextExpression;
    }

    /**
     * Resets expression.
     *
     */
    public void reset() {
        if (nextExpression != null) nextExpression.reset();
    }

    /**
     * Sets previous expression for gradient calculation chain.
     *
     * @param previousExpression previous expression.
     */
    public void setPreviousExpression(AbstractExpression previousExpression) {
        this.previousExpression = previousExpression;
    }

    /**
     * Calculates entire expression step including normalization and regulation.
     *
     * @param index index
     * @param firstKey first key of inputs
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateExpressionStep(int index, int firstKey) throws MatrixException, DynamicParamException {
        calculateExpressionStep(index == firstKey, index);
        if (nextExpression != null) nextExpression.calculateExpressionStep(index, firstKey);
    }

    /**
     * Calculates entire expression step including normalization and regulation.
     *
     * @param indices indices
     * @param firstKey first key of inputs
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateExpressionStep(Set<Integer> indices, int firstKey) throws MatrixException, DynamicParamException {
        for (Integer index : indices) {
            calculateExpressionStep(index == firstKey, index);
        }
        if (nextExpression != null) nextExpression.calculateExpressionStep(indices, firstKey);
    }

    /**
     * Calculates entire expression step including normalization and regulation.
     *
     * @param firstCalculateExpressionStep true if this is first calculation step for expression
     * @param index index
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateExpressionStep(boolean firstCalculateExpressionStep, int index) throws MatrixException, DynamicParamException {
        updateExpressionDependency(index);
        if (firstCalculateExpressionStep) forwardRegularize();
        if (firstCalculateExpressionStep) forwardNormalize();
        forwardNormalize(index);
        if (firstCalculateExpressionStep) calculateExpression();
        calculateExpression(index);
        if (firstCalculateExpressionStep) forwardNormalizeFinalize();
    }

    /**
     * Updates expression forward direction dependency.
     *
     * @param index index
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching or node is of type multi-index.
     */
    protected void updateExpressionDependency(int index) throws MatrixException {
        argument1.updateMatrixDependency(index);
    }

    /**
     * Calculates expression.
     *
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected abstract void calculateExpression() throws MatrixException, DynamicParamException;

    /**
     * Calculates expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation fails.
     */
    protected abstract void calculateExpression(int index) throws MatrixException;

    /**
     * Calculates entire gradient step including normalization and regulation.
     *
     * @param index index
     * @param lastKey last key of inputs
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateGradientStep(int index, int lastKey) throws MatrixException, DynamicParamException {
        calculateGradientStep (index == lastKey, index);
        if (previousExpression != null) previousExpression.calculateGradientStep(index, lastKey);
    }

    /**
     * Calculates entire gradient step including normalization and regulation.
     *
     * @param indices indices
     * @param lastKey last key of inputs
     * @param steps number of gradient steps taken
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateGradientStep(Set<Integer> indices, int lastKey, int steps) throws MatrixException, DynamicParamException {
        int step = 0;
        for (Integer index : indices) {
            boolean stopAtStep = steps > 0 && ++step >= steps;
            calculateGradientStep (index == lastKey || stopAtStep, index);
            if (stopAtStep) break;
        }
        if (previousExpression != null) previousExpression.calculateGradientStep(indices, lastKey, steps);
    }

    /**
     * Calculates gradient step including normalization and regulation.
     *
     * @param lastCalculateGradientStep true if this is last gradient step for expression
     * @param index index
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void calculateGradientStep(boolean lastCalculateGradientStep, int index) throws MatrixException, DynamicParamException {
        updateGradientDependency(index);
        if (lastCalculateGradientStep) calculateGradient();
        calculateGradient(index);
        backwardNormalize(index);
        if (lastCalculateGradientStep) backwardNormalize();
        if (lastCalculateGradientStep) backwardRegularize();
    }

    /**
     * Updates gradient dependency to backward direction.
     *
     * @param index index
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching or node is of type multi-index.
     */
    protected void updateGradientDependency(int index) throws MatrixException {
        result.updateGradientDependency(index);
    }

    /**
     * Calculates gradient of expression.
     *
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    protected abstract void calculateGradient() throws MatrixException;

    /**
     * Calculates gradient of expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation of gradient fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected abstract void calculateGradient(int index) throws MatrixException, DynamicParamException;

    /**
     * Execute forward normalization to constant node.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void forwardNormalize() throws MatrixException, DynamicParamException {
        argument1.forwardNormalize();
    }

    /**
     * Execute forward normalization to specific entry (sample)
     *
     * @param sampleIndex sample index of specific entry.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void forwardNormalize(int sampleIndex) throws MatrixException, DynamicParamException {
        argument1.forwardNormalize(sampleIndex);
    }

    /**
     * Execute forward normalization finalize callback to constant node.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void forwardNormalizeFinalize() throws MatrixException {
        argument1.forwardNormalizeFinalize();
    }

    /**
     * Execute backward normalization to all entries of node.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void backwardNormalize() throws MatrixException, DynamicParamException {
        argument1.backwardNormalize();
    }

    /**
     * Execute backward normalization to specific entry (sample)
     *
     * @param sampleIndex sample index of specific entry.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void backwardNormalize(int sampleIndex) throws MatrixException, DynamicParamException {
        argument1.backwardNormalize(sampleIndex);
    }

    /**
     * Executes forward regularization step .
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void forwardRegularize() throws MatrixException {
        argument1.forwardRegularize();
    }

    /**
     * Cumulates error from regularization.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return updated error value.
     */
    public double cumulateRegularizationError() throws MatrixException, DynamicParamException {
        return argument1.cumulateRegularizationError() + (nextExpression != null ? nextExpression.cumulateRegularizationError() : 0);
    }

    /**
     * Executes backward regularization.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void backwardRegularize() throws MatrixException {
        argument1.backwardRegularize();
    }

    /**
     * Prints expression chain.
     *
     */
    public void printExpressionChain() {
        System.out.println("Chain of expressions for procedure: ");
        invokePrintExpressionChain();
    }

    /**
     * Prints expression chain.
     *
     */
    private void invokePrintExpressionChain() {
        printExpression();
        if (nextExpression != null) nextExpression.invokePrintExpressionChain();
    }

    /**
     * Prints gradient chain.
     *
     */
    public void printGradientChain() {
        System.out.println("Chain of gradients for procedure: ");
        invokePrintGradientChain();
    }

    /**
     * Prints gradient chain.
     *
     */
    private void invokePrintGradientChain() {
        printGradient();
        if (previousExpression != null) previousExpression.invokePrintGradientChain();
    }

    /**
     * Prints expression.
     *
     */
    public abstract void printExpression();

    /**
     * Prints gradient.
     *
     */
    public abstract void printGradient();

    /**
     * Prints expression.
     *
     */
    protected void print() {
        System.out.print("Expression " +getExpressionID() + ": ");
    }

    /**
     * Returns gradient identifier name.
     *
     * @return gradient identifier name.
     */
    @SuppressWarnings("SameReturnValue")
    protected String getGradientIdentifierName() {
        return "d";
    }

    /**
     * Return gradient name of argument1.
     *
     * @return gradient name of argument1.
     */
    protected String getArgument1GradientName() {
        return getGradientIdentifierName() + argument1.getName();
    }

    /**
     * Return gradient name of result.
     *
     * @return gradient name of result.
     */
    protected String getResultGradientName() {
        return getGradientIdentifierName() + result.getName();
    }

    /**
     * Returns gradient prefix for argument1.
     *
     * @return gradient prefix for argument1.
     */
    protected String getArgument1PrefixName() {
        return getName() + ": " + getArgument1GradientName() + " = " + "" + getArgument1SumPrefix();
    }

    /**
     * Returns argument1 prefix.
     *
     * @return argument1 prefix.
     */
    protected String getArgument1SumPrefix() {
        return !argument1.isMultiIndex() ? "sum(" : "";
    }

    /**
     * Returns argument1 postfix.
     *
     * @return argument1 postfix.
     */
    protected String getArgument1SumPostfix() {
        return !argument1.isMultiIndex() ? ")" : "";
    }

    /**
     * Returns node gradient name.
     *
     * @param node node
     * @return node gradient name.
     */
    protected String getNodeGradientName(Node node) {
        return getGradientIdentifierName() + node.getName();
    }

    /**
     * Returns node prefix name.
     *
     * @param node node
     * @param negateResult if true result will be negated.
     * @return node prefix name.
     */
    protected String getNodePrefixName(Node node, boolean negateResult) {
        return getName() + ": " + getNodeGradientName(node) + " = " + (negateResult ? "-" : "") + getNodeSumPrefix(node);
    }

    /**
     * Returns node prefix name with result.
     *
     * @param node node
     * @param negateResult if true result will be negated.
     * @return node prefix name with result.
     */
    protected String getNodeWithResultPrefixName(Node node, boolean negateResult) {
        return getNodePrefixName(node, negateResult) + getResultGradientName();
    }

    /**
     * Returns node sum prefix.
     *
     * @param node node.
     * @return node sum prefix.
     */
    protected String getNodeSumPrefix(Node node) {
        return !node.isMultiIndex() ? "sum(" : "";
    }

    /**
     * Returns node sum postfix.
     *
     * @param node node.
     * @return node sum postfix.
     */
    protected String getNodeSumPostfix(Node node) {
        return !node.isMultiIndex() ? ")" : "";
    }

    /**
     * Prints gradient for argument1
     *  @param withResultPrefix true if result prefix is added.
     * @param suffix suffix part for gradient expression.
     */
    protected void printArgument1Gradient(boolean withResultPrefix, String suffix) {
        print();
        System.out.println((withResultPrefix ? getNodeWithResultPrefixName(argument1, false) : getNodePrefixName(argument1, false)) + (suffix != null ? suffix : "") + getNodeSumPostfix(argument1));
    }

}
