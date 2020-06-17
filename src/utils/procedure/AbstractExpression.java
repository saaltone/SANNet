/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

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
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public AbstractExpression(int expressionID, Node argument1, Node result) throws MatrixException {
        this.expressionID = expressionID;
        if (argument1 == null) throw new MatrixException("First argument not defined.");
        this.argument1 = argument1;
        this.result = result;
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
     */
    public void calculateExpressionStep(int index, int firstKey) throws MatrixException {
        calculateExpressionStep(index == firstKey, index);
        if (nextExpression != null) nextExpression.calculateExpressionStep(index, firstKey);
    }

    /**
     * Calculates entire expression step including normalization and regulation.
     *
     * @param indices indices
     * @param firstKey first key of inputs
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpressionStep(Set<Integer> indices, int firstKey) throws MatrixException {
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
     */
    public void calculateExpressionStep(boolean firstCalculateExpressionStep, int index) throws MatrixException {
        if (firstCalculateExpressionStep) forwardRegularize();
        if (firstCalculateExpressionStep) forwardNormalize();
        forwardNormalize(index);
        if (firstCalculateExpressionStep) calculateExpression();
        calculateExpression(index);
        if (firstCalculateExpressionStep) forwardNormalizeFinalize();
    }

    /**
     * Calculates expression.
     *
     * @throws MatrixException throws exception if calculation fails.
     */
    protected abstract void calculateExpression() throws MatrixException;

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
     */
    public void calculateGradientStep(int index, int lastKey) throws MatrixException {
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
     */
    public void calculateGradientStep(Set<Integer> indices, int lastKey, int steps) throws MatrixException {
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
     */
    private void calculateGradientStep(boolean lastCalculateGradientStep, int index) throws MatrixException {
        if (lastCalculateGradientStep) calculateGradient();
        calculateGradient(index);
        backwardNormalize(index);
        if (lastCalculateGradientStep) backwardNormalize();
        if (lastCalculateGradientStep) backwardRegularize();
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
     */
    protected abstract void calculateGradient(int index) throws MatrixException;

    /**
     * Execute forward normalization to constant node.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void forwardNormalize() throws MatrixException {
        argument1.forwardNormalize();
    }

    /**
     * Execute forward normalization to specific entry (sample)
     *
     * @param sampleIndex sample index of specific entry.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void forwardNormalize(int sampleIndex) throws MatrixException {
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
     */
    protected void backwardNormalize() throws MatrixException {
        argument1.backwardNormalize();
    }

    /**
     * Execute backward normalization to specific entry (sample)
     *
     * @param sampleIndex sample index of specific entry.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void backwardNormalize(int sampleIndex) throws MatrixException {
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
     * @throws MatrixException throws exception if matrix operation fails.
     * @return updated error value.
     */
    public double cumulateRegularizationError() throws MatrixException {
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

}
