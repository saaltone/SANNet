/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;
import utils.procedure.node.Node;

import java.io.Serial;
import java.io.Serializable;
import java.util.Set;

/**
 * Implements single computable expression including gradient expression.<br>
 * Assumes underlying class that implements specific expression.<br>
 *
 */
@SuppressWarnings("SameReturnValue")
public abstract class AbstractExpression implements Expression, Serializable {

    @Serial
    private static final long serialVersionUID = -3692842009210981254L;

    /**
     * Name of expression;
     *
     */
    private final String expressionName;

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
     * Next expression for expression calculation.
     *
     */
    private Expression nextExpression;

    /**
     * Previous expression for gradient calculation.
     *
     */
    private Expression previousExpression;

    /**
     * If true expression is active otherwise not-active.
     *
     */
    private boolean isActive = true;

    /**
     * Constructor for abstract expression.
     *
     * @param expressionName     name of expression.
     * @param operationSignature operation signature of expression.
     * @param expressionID       unique ID for expression.
     * @param argument1          first argument.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public AbstractExpression(String expressionName, String operationSignature, int expressionID, Node argument1) throws MatrixException {
        if (argument1 == null) throw new MatrixException("First argument not defined.");
        this.expressionName = expressionName;
        this.operationSignature = operationSignature;
        this.expressionID = expressionID;
    }

    /**
     * Returns name of expression.
     *
     * @return name of expression.
     */
    public String getExpressionName() {
        return expressionName;
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
     * Checks if result gradient is defined for specific sample index.
     *
     * @param result result
     * @param sampleIndex sample index.
     * @throws MatrixException throws exception if argument is not defined.
     */
    protected void checkResultGradient(Node result, int sampleIndex) throws MatrixException {
        if (result.getGradient(sampleIndex) == null) throw new MatrixException(getExpressionName() + ": Result gradient not defined for sample index" + sampleIndex);
    }

    /**
     * Sets next expression for expression calculation chain.
     *
     * @param nextExpression next expression.
     */
    public void setNextExpression(Expression nextExpression) {
        this.nextExpression = nextExpression;
    }

    /**
     * Sets previous expression for gradient calculation chain.
     *
     * @param previousExpression previous expression.
     */
    public void setPreviousExpression (Expression previousExpression) {
        this.previousExpression = previousExpression;
    }

    /**
     * Returns true is expression is executed as single step otherwise false.
     *
     * @return true is expression is executed as single step otherwise false.
     */
    protected abstract boolean executeAsSingleStep();

    /**
     * Resets expression.
     *
     */
    public void reset() {
        applyReset();
        if (nextExpression != null) nextExpression.reset();
    }

    /**
     * Resets expression.
     *
     */
    protected abstract void applyReset();

    /**
     * Sets is expression is active.
     *
     * @param isActive is true expression is active otherwise non-active.
     */
    public void setActive(boolean isActive) {
        this.isActive = isActive;
        if (nextExpression != null) nextExpression.setActive(isActive);
    }

    /**
     * Returns is expression is active.
     *
     * @return returns true if expression is active otherwise false.
     */
    protected boolean isActive() {
        return isActive;
    }

    /**
     * Calculates entire expression chain including regulation.
     *
     * @param sampleIndex sample index
     * @param firstSampleIndex first sample index
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateExpressionStep(int sampleIndex, int firstSampleIndex) throws MatrixException, DynamicParamException {
        if (executeAsSingleStep() && sampleIndex == firstSampleIndex) calculateExpression();
        else calculateExpression(sampleIndex);
        if (nextExpression != null) nextExpression.calculateExpressionStep(sampleIndex, firstSampleIndex);
    }

    /**
     * Calculates entire expression chain including regulation.
     *
     * @param sampleIndices sample indices
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateExpressionStep(Set<Integer> sampleIndices) throws MatrixException, DynamicParamException {
        if (executeAsSingleStep()) calculateExpression();
        else for (Integer sampleIndex : sampleIndices) calculateExpression(sampleIndex);
        if (nextExpression != null) nextExpression.calculateExpressionStep(sampleIndices);
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
     * @param sampleIndex sample index.
     * @throws MatrixException throws exception if calculation fails.
     */
    protected abstract void calculateExpression(int sampleIndex) throws MatrixException;

    /**
     * Calculates entire gradient expression chain including regulation.
     *
     * @param sampleIndex sample index
     * @param lastSampleIndex last sample index
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateGradientStep(int sampleIndex, int lastSampleIndex) throws MatrixException, DynamicParamException {
        if (executeAsSingleStep() && sampleIndex == lastSampleIndex) calculateGradient();
        else calculateGradient(sampleIndex);
        if (previousExpression != null) previousExpression.calculateGradientStep(sampleIndex, lastSampleIndex);
    }

    /**
     * Calculates entire gradient expression chain including regulation.
     *
     * @param sampleIndices sample indices
     * @param numberOfGradientSteps number of gradient steps taken
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateGradientStep(Set<Integer> sampleIndices, int numberOfGradientSteps) throws MatrixException, DynamicParamException {
        if (executeAsSingleStep()) calculateGradient();
        else {
            int gradientStepCount = 0;
            for (Integer sampleIndex : sampleIndices) {
                calculateGradient(sampleIndex);
                if (numberOfGradientSteps > 0 && ++gradientStepCount >= numberOfGradientSteps) break;
            }
        }
        if (previousExpression != null) previousExpression.calculateGradientStep(sampleIndices, numberOfGradientSteps);
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
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    protected abstract void calculateGradient(int sampleIndex) throws MatrixException;

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
    public void invokePrintExpressionChain() {
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
    public void invokePrintGradientChain() {
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

}
