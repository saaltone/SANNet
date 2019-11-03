package utils.procedure;

import utils.matrix.MatrixException;

/**
 * Class that defined unary expression.
 *
 */
public abstract class AbstractUnaryExpression extends AbstractExpression {

    /**
     * Constructor for unary expression.
     *
     * @param expressionID expression ID
     * @param arg1 first argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    AbstractUnaryExpression(int expressionID, Node arg1, Node result) throws MatrixException {
        super(expressionID, arg1, result);
    }

}
