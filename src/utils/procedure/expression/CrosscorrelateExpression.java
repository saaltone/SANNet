/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.matrix.operation.CrosscorrelationFilterGradientMatrixOperation;
import utils.matrix.operation.CrosscorrelationInputGradientMatrixOperation;
import utils.matrix.operation.CrosscorrelationMatrixOperation;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Implements expression for crosscorrelation operation.<br>
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
     * Returns true is expression is executed as single step otherwise false.
     *
     * @return true is expression is executed as single step otherwise false.
     */
    protected boolean executeAsSingleStep() {
        return false;
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
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int sampleIndex) throws MatrixException {
        if (argument1.getMatrix(sampleIndex) == null || argument2.getMatrix(sampleIndex) == null) throw new MatrixException(getExpressionName() + ": Arguments for operation not defined");
        crosscorrelationMatrixOperation.apply(argument1.getMatrix(sampleIndex), argument2.getMatrix(sampleIndex), result.getNewMatrix(sampleIndex));
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
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int sampleIndex) throws MatrixException {
        if (result.getGradient(sampleIndex) == null) throw new MatrixException(getExpressionName() + ": Result gradient not defined.");
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, crosscorrelationInputGradientMatrixOperation.apply(result.getGradient(sampleIndex), argument2.getMatrix(sampleIndex), argument1.getNewMatrix()), false);
        if (!argument2.isStopGradient()) argument2.cumulateGradient(sampleIndex, crosscorrelationFilterGradientMatrixOperation.apply(result.getGradient(sampleIndex), argument1.getMatrix(sampleIndex), argument2.getNewMatrix()), false);
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
