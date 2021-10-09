/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.matrix.operation.CrosscorrelationFilterGradientMatrixOperation;
import utils.matrix.operation.CrosscorrelationInputGradientMatrixOperation;
import utils.matrix.operation.CrosscorrelationMatrixOperation;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Class that describes expression for crosscorrelation operation.<br>
 *
 */
public class CrosscorrelateExpression extends AbstractBinaryExpression implements Serializable {

    /**
     * Reference to crosscorrelation matrix operation.
     *
     */
    private final CrosscorrelationMatrixOperation crosscorrelationMatrixOperation;

    /**
     * Reference to crosscorrelation input gradient matrix operation.
     *
     */
    private final CrosscorrelationInputGradientMatrixOperation crosscorrelationInputGradientMatrixOperation;

    /**
     * Reference to crosscorrelation filter gradient matrix operation.
     *
     */
    private final CrosscorrelationFilterGradientMatrixOperation crosscorrelationFilterGradientMatrixOperation;

    /**
     * Constructor for crosscorrelation operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of expression.
     * @param stride stride of crosscorrelation operation.
     * @param dilation dilation step size for crosscorrelation operation.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public CrosscorrelateExpression(int expressionID, Node argument1, Node argument2, Node result, int stride, int dilation) throws MatrixException {
        super("CROSSCORRELATE", "CROSSCORRELATE", expressionID, argument1, argument2, result);
        crosscorrelationMatrixOperation = new CrosscorrelationMatrixOperation(result.getRows(), result.getColumns(), argument2.getRows(), argument2.getColumns(), dilation, stride);
        crosscorrelationInputGradientMatrixOperation = new CrosscorrelationInputGradientMatrixOperation(result.getRows(), result.getColumns(), argument2.getRows(), argument2.getColumns(), dilation, stride);
        crosscorrelationFilterGradientMatrixOperation = new CrosscorrelationFilterGradientMatrixOperation(result.getRows(), result.getColumns(), argument2.getRows(), argument2.getColumns(), dilation, stride);
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
        if (argument1.getMatrix(index) == null || argument2.getMatrix(index) == null) throw new MatrixException(getExpressionName() + ": Arguments for operation not defined");
        crosscorrelationMatrixOperation.apply(argument1.getMatrix(index), argument2.getMatrix(index), result.getNewMatrix(index));
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
        if (!argument1.isStopGradient()) argument1.cumulateGradient(index, crosscorrelationInputGradientMatrixOperation.apply(result.getGradient(index), argument2.getMatrix(index), argument1.getEmptyMatrix()), false);
        if (!argument2.isStopGradient()) argument2.cumulateGradient(index, crosscorrelationFilterGradientMatrixOperation.apply(result.getGradient(index), argument1.getMatrix(index), argument2.getEmptyMatrix()), false);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        printSpecificBinaryExpression();
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(false, getExpressionName() + "_GRADIENT(d" + result.getName() + ", " + argument2.getName() + ")");
        printArgument2Gradient(false, false, getExpressionName() + "_GRADIENT(d" + result.getName() + ", " + argument1.getName() + ")");
    }

}
