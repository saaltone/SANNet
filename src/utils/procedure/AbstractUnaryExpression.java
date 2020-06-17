/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

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
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    AbstractUnaryExpression(int expressionID, Node argument1, Node result) throws MatrixException {
        super(expressionID, argument1, result);
    }

}
