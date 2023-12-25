/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.configurable.DynamicParamException;
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
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public WinogradConvolutionExpression(int expressionID, Node argument1, Node argument2, Node result, int stride, int dilation) throws MatrixException, DynamicParamException {
        super("WINOGRAD_CONVOLUTION", expressionID, argument1, argument2, result);

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
     * Calculates result matrix.
     *
     * @return result matrix.
     */
    protected Matrix calculateResult() {
        return null;
    }

    /**
     * Calculates result matrix.
     *
     * @param sampleIndex sample index
     * @param argument1Matrix argument1 matrix for a sample index.
     * @param argument2Matrix argument2 matrix for a sample index.
     * @return result matrix.
     * @throws MatrixException throws exception if calculation fails.
     */
    protected Matrix calculateResult(int sampleIndex, Matrix argument1Matrix, Matrix argument2Matrix) throws MatrixException {
        Matrix preprocessedFilter = G.dot(argument2.getMatrix(sampleIndex)).dot(GT);
        return winogradConvolutionMatrixOperation.apply(argument1.getMatrix(sampleIndex), preprocessedFilter);
    }

    /**
     * Calculates argument 1 gradient matrix.
     */
    protected void calculateArgument1Gradient() {
    }

    /**
     * Calculates argument 1 gradient matrix.
     *
     * @param sampleIndex     sample index.
     * @param resultGradient  result gradient.
     * @param argument1Matrix argument 1 matrix.
     * @param argument2Matrix argument 2 matrix.
     * @param resultMatrix    result matrix.
     * @return argument1 gradient matrix.
     * @throws MatrixException throws exception if calculation fails.
     */
    protected Matrix calculateArgument1Gradient(int sampleIndex, Matrix resultGradient, Matrix argument1Matrix, Matrix argument2Matrix, Matrix resultMatrix) throws MatrixException {
        return crosscorrelationInputGradientMatrixOperation.apply(resultGradient, argument2Matrix);
    }

    /**
     * Calculates argument 2 gradient matrix.
     *
     * @param sampleIndex     sample index.
     * @param resultGradient  result gradient.
     * @param argument1Matrix argument 1 matrix.
     * @param argument2Matrix argument 2 matrix.
     * @param resultMatrix    result matrix.
     * @return argument1 gradient matrix.
     * @throws MatrixException throws exception if calculation fails.
     */
    protected Matrix calculateArgument2Gradient(int sampleIndex, Matrix resultGradient, Matrix argument1Matrix, Matrix argument2Matrix, Matrix resultMatrix) throws MatrixException {
        return crosscorrelationFilterGradientMatrixOperation.apply(resultGradient, argument1Matrix);
    }

    /**
     * Returns expression operation signature.
     *
     * @return expression operation signature.
     */
    protected String getExpressionOperationSignature() {
        return getExpressionName() + "(" + getArgument1().getName() + ", " + getArgument2().getName() + ")";
    }

    /**
     * Returns gradient 1 operation signature.
     *
     * @return gradient 1 operation signature.
     */
    protected String getGradientOperation1Signature() {
        return getExpressionName() + "_INPUT_GRADIENT(d" + getResult().getName() + ", " + getArgument2().getName() + ")";
    }

    /**
     * Returns gradient 2 operation signature.
     *
     * @return gradient 2 operation signature.
     */
    protected String getGradientOperation2Signature() {
        return getExpressionName() + "_FILTER_GRADIENT(d" + getResult().getName() + ", " + getArgument1().getName() + ")";
    }

}
