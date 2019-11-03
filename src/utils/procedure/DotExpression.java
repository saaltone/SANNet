/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import utils.matrix.MatrixException;

import java.io.Serializable;

/**
 * Class that describes expression for dot operation.
 *
 */
public class DotExpression extends AbstractBinaryExpression implements Serializable {

    /**
     * Constructor for dot operation.
     *
     * @param expressionID unique ID for expression.
     * @param arg1 first argument.
     * @param arg2 second argument.
     * @param result result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public DotExpression(int expressionID, Node arg1, Node arg2, Node result) throws MatrixException {
        super(expressionID, arg1, arg2, result);
    }

    /**
     * Calculates expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int index) throws MatrixException {
        if (arg1.getMatrix(index) == null || arg2.getMatrix(index) == null) throw new MatrixException("Arguments for DOT operation not defined");
        result.setMatrix(index, arg1.getMatrix(index).dot(arg2.getMatrix(index)));
    }

    /**
     * Calculates gradient of expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int index) throws MatrixException {
        if (result.getGradient(index) == null) throw new MatrixException("Result gradient not defined.");
        arg1.updateGradient(index, result.getGradient(index).dot(arg2.getMatrix(index).T()), true);
        arg2.updateGradient(index, arg1.getMatrix(index).T().dot(result.getGradient(index)), true);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        System.out.print("DOT: " + arg1 + " " + arg2 + " " + result);
    }

}
