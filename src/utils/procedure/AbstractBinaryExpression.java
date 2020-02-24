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
    protected Node arg2;

    /**
     * Constructor for binary expression.
     *
     * @param expressionID expression ID
     * @param arg1 first argument.
     * @param arg2 second argument.
     * @param result result of node.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    AbstractBinaryExpression(int expressionID, Node arg1, Node arg2, Node result) throws MatrixException {
        super(expressionID, arg1, result);
        if (arg2 == null) throw new MatrixException("Second argument not defined.");
        this.arg2 = arg2;
    }

    /**
     * Returns second argument of expression.
     *
     * @return second argument of expression.
     */
    public Node getArg2() {
        return arg2;
    }

    /**
     * Resets nodes of expression.
     *
     */
    public void resetExpression() {
        super.resetExpression();
        arg2.resetNode();
    }

    /**
     * Resets nodes of expression for specific data index.
     *
     * @param index data index.
     */
    public void resetExpression(int index) throws MatrixException {
        super.resetExpression(index);
        arg2.resetNode(index);
    }

    /**
     * Make forward callback to all entries of node.
     *
     * @throws MatrixException throws exception is matrix operation fails.
     */
    public void forwardCallback() throws MatrixException {
        super.forwardCallback();
        arg2.forwardCallback();
    }

    /**
     * Make forward callback to specific entry (sample)
     *
     * @param sampleIndex sample index of specific entry.
     * @throws MatrixException throws exception is matrix operation fails.
     */
    public void forwardCallback(int sampleIndex) throws MatrixException {
        super.forwardCallback(sampleIndex);
        arg2.forwardCallback(sampleIndex);
    }

    /**
     * Make backward callback to all entries of node.
     *
     * @throws MatrixException throws exception is matrix operation fails.
     */
    public void backwardCallback() throws MatrixException {
        super.backwardCallback();
        arg2.backwardCallback();
    }

    /**
     * Make backward callback to specific entry (sample)
     *
     * @param sampleIndex sample index of specific entry.
     * @throws MatrixException throws exception is matrix operation fails.
     */
    public void backwardCallback(int sampleIndex) throws MatrixException {
        super.backwardCallback(sampleIndex);
        arg2.backwardCallback(sampleIndex);
    }

}
