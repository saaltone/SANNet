/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.operation.CrosscorrelationFilterGradientMatrixOperation;
import utils.matrix.operation.CrosscorrelationInputGradientMatrixOperation;
import utils.matrix.operation.WinogradConvolutionMatrixOperation;
import utils.procedure.node.Node;

/**
 * Implements expression for Winograd convolution operation.<br>
 *
 */
public class WinogradConvolutionExpression extends AbstractBinaryExpression {

    /**
     * G matrix for Winograd convolution.
     *
     */
    private final Matrix G;

    /**
     * G transposed matrix for Winograd convolution.
     *
     */
    private final Matrix GT;

    /**
     * Preprocessed filter.
     *
     */
    private Matrix preprocessedFilter;

    /**
     * Reference to crosscorrelation matrix operation.
     *
     */
    private final WinogradConvolutionMatrixOperation winogradConvolutionMatrixOperation;

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
     * Constructor for Winograd convolution operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of expression.
     * @param stride stride of crosscorrelation operation.
     * @param dilation dilation step size for crosscorrelation operation.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public WinogradConvolutionExpression(int expressionID, Node argument1, Node argument2, Node result, int stride, int dilation) throws MatrixException {
        super("WINOGRAD_CONVOLUTION", "WINOGRAD_CONVOLUTION", expressionID, argument1, argument2, result);

        Matrix AT = WinogradConvolutionMatrixOperation.getATMatrix(result.getDepth());
        Matrix a = AT.transpose().copy(true);

        Matrix c = WinogradConvolutionMatrixOperation.getCMatrix(result.getDepth());
        Matrix CT = c.transpose().copy(true);

        G = WinogradConvolutionMatrixOperation.getGMatrix(result.getDepth());
        GT = G.transpose().copy(true);

        winogradConvolutionMatrixOperation = new WinogradConvolutionMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), a, AT, c, CT);
        crosscorrelationInputGradientMatrixOperation = new CrosscorrelationInputGradientMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), argument1.getDepth(), argument2.getRows(), argument2.getColumns(), dilation, stride, false);
        crosscorrelationFilterGradientMatrixOperation = new CrosscorrelationFilterGradientMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), argument1.getDepth(), argument2.getRows(), argument2.getColumns(), dilation, stride, false);
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
     * Resets expression.
     *
     */
    public void applyReset() {
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
        checkArguments(argument1, argument2, sampleIndex);
        if (preprocessedFilter == null) preprocessedFilter = G.dot(argument2.getMatrix(sampleIndex)).dot(GT);
        result.setMatrix(sampleIndex, winogradConvolutionMatrixOperation.apply(argument1.getMatrix(sampleIndex), preprocessedFilter));
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
        checkResultGradient(result, sampleIndex);
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, crosscorrelationInputGradientMatrixOperation.apply(result.getGradient(sampleIndex), argument2.getMatrix(sampleIndex)), false);
        if (!argument2.isStopGradient()) argument2.cumulateGradient(sampleIndex, crosscorrelationFilterGradientMatrixOperation.apply(result.getGradient(sampleIndex), argument1.getMatrix(sampleIndex)), false);
        preprocessedFilter = null;
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
