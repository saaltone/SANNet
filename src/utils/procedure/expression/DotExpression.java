/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.matrix.operation.DotMatrixOperation;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Class that describes expression for dot operation.<br>
 *
 */
public class DotExpression extends AbstractBinaryExpression implements Serializable {

    /**
     * Reference to dot matrix operation.
     *
     */
    private final DotMatrixOperation dotMatrixOperation;

    /**
     * Reference to dot gradient 1 matrix operation.
     *
     */
    private final DotMatrixOperation dotGradient1MatrixOperation;

    /**
     * Reference to dot gradient 2 matrix operation.
     *
     */
    private final DotMatrixOperation dotGradient2MatrixOperation;

    /**
     * Constructor for dot operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public DotExpression(int expressionID, Node argument1, Node argument2, Node result) throws MatrixException {
        super("DOT", "x", expressionID, argument1, argument2, result);
        dotMatrixOperation = new DotMatrixOperation(argument1.getRows(), argument2.getColumns());
        dotGradient1MatrixOperation = new DotMatrixOperation(result.getRows(), argument2.getRows());
        dotGradient2MatrixOperation = new DotMatrixOperation(argument1.getColumns(), result.getColumns());
    }

    /**
     * Calculates expression.
     *
     */
    public void calculateExpression() {
    }

    /**
     * Calculates expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int index) throws MatrixException {
        if (argument1.getMatrix(index) == null || argument2.getMatrix(index) == null) throw new MatrixException(getExpressionName() + "Arguments for operation not defined");
        dotMatrixOperation.apply(argument1.getMatrix(index), argument2.getMatrix(index), result.getNewMatrix(index));
    }

    /**
     * Calculates gradient of expression.
     *
     */
    public void calculateGradient() {
    }

    /**
     * Calculates gradient of expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int index) throws MatrixException {
        if (result.getGradient(index) == null) throw new MatrixException(getExpressionName() + ": Result gradient not defined.");
        if (!argument1.isStopGradient()) argument1.cumulateGradient(index, dotGradient1MatrixOperation.apply(result.getGradient(index), argument2.getMatrix(index).transpose(), argument1.getEmptyMatrix()), false);
        if (!argument2.isStopGradient()) argument2.cumulateGradient(index, dotGradient2MatrixOperation.apply(argument1.getMatrix(index).transpose(), result.getGradient(index), argument2.getEmptyMatrix()), false);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        printBasicBinaryExpression();
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(true, " " + getOperationSignature() + " " + argument2.getName() + ".T");
        printArgument2Gradient(false, false, argument1.getName() + ".T" + " " + getOperationSignature() + " " + getResultGradientName());
    }

}
