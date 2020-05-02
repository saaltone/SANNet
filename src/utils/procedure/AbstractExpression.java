/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import utils.matrix.MatrixException;

import java.io.Serializable;

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
    protected Node arg1;

    /**
     * Node for result.
     *
     */
    protected Node result;

    /**
     * Constructor for abstract expression.
     *
     * @param expressionID unique ID for expression.
     * @param arg1 first argument.
     * @param result result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public AbstractExpression(int expressionID, Node arg1, Node result) throws MatrixException {
        this.expressionID = expressionID;
        if (arg1 == null) throw new MatrixException("First argument not defined.");
        this.arg1 = arg1;
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
    public Node getArg1() {
        return arg1;
    }

    /**
     * Returns second argument of expression.
     *
     * @return returns null unless overloaded by abstract binary expression class.
     */
    public Node getArg2() {
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
     * Resets nodes of expression.
     *
     */
    public void resetExpression() {
        arg1.resetNode();
        if (result != null) result.resetNode();
    }

    /**
     * Resets nodes of expression for specific data index.
     *
     * @param index data index.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void resetExpression(int index) throws MatrixException {
        arg1.resetNode(index);
        if (result != null) result.resetNode(index);
    }

    /**
     * Make forward callback to all entries of node.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardCallback() throws MatrixException {
        arg1.forwardCallback();
    }

    /**
     * Make forward callback to specific entry (sample)
     *
     * @param sampleIndex sample index of specific entry.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardCallback(int sampleIndex) throws MatrixException {
        arg1.forwardCallback(sampleIndex);
    }

    /**
     * Make backward callback to all entries of node.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardCallback() throws MatrixException {
        arg1.backwardCallback();
    }

    /**
     * Make backward callback to specific entry (sample)
     *
     * @param sampleIndex sample index of specific entry.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardCallback(int sampleIndex) throws MatrixException {
        arg1.backwardCallback(sampleIndex);
    }

    /**
     * Calculates expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public abstract void calculateExpression(int index) throws MatrixException;

    /**
     * Calculates gradient of expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public abstract void calculateGradient(int index) throws MatrixException;

    /**
     * Prints expression.
     *
     */
    public abstract void printExpression();

}
