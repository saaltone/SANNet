/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import utils.matrix.MatrixException;

/**
 * Class that defines binary expression.
 *
 */
public abstract class AbstractBinaryExpression extends AbstractExpression {

    /**
     * Node for second argument.
     *
     */
    protected final Node argument2;

    /**
     * Constructor for binary expression.
     *
     * @param expressionID expression ID
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of node.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    AbstractBinaryExpression(int expressionID, Node argument1, Node argument2, Node result) throws MatrixException {
        super(expressionID, argument1, result);
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
     * Make forward callback to all entries of node.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardNormalize() throws MatrixException {
        super.forwardNormalize();
        argument2.forwardNormalize();
    }

    /**
     * Make forward callback to specific entry (sample)
     *
     * @param sampleIndex sample index of specific entry.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardNormalize(int sampleIndex) throws MatrixException {
        super.forwardNormalize(sampleIndex);
        argument2.forwardNormalize(sampleIndex);
    }

    /**
     * Make backward callback to all entries of node.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardNormalize() throws MatrixException {
        super.backwardNormalize();
        argument2.backwardNormalize();
    }

    /**
     * Make backward callback to specific entry (sample)
     *
     * @param sampleIndex sample index of specific entry.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardNormalize(int sampleIndex) throws MatrixException {
        super.backwardNormalize(sampleIndex);
        argument2.backwardNormalize(sampleIndex);
    }

    /**
     * Executes forward regularization step .
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
     * @throws MatrixException throws exception if matrix operation fails.
     * @return updated error value.
     */
    public double cumulateRegularizationError() throws MatrixException {
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

}
