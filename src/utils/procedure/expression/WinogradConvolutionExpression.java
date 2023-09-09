/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.DMatrix;
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

        Matrix AT = new DMatrix(2, 4, 1);
        AT.setValue(0, 0, 0, 1);
        AT.setValue(0, 1, 0, 1);
        AT.setValue(0, 2, 0, 1);
        AT.setValue(0, 3, 0, 0);
        AT.setValue(1, 0, 0, 0);
        AT.setValue(1, 1, 0, 1);
        AT.setValue(1, 2, 0, -1);
        AT.setValue(1, 3, 0, -1);
        maskZeros(AT);
        Matrix a = AT.transpose();

        Matrix c = new DMatrix(4, 4, 1);
        c.setValue(0, 0, 0, 1);
        c.setValue(0, 1, 0, 0);
        c.setValue(0, 2, 0, -1);
        c.setValue(0, 3, 0, 0);
        c.setValue(1, 0, 0, 0);
        c.setValue(1, 1, 0, 1);
        c.setValue(1, 2, 0, 1);
        c.setValue(1, 3, 0, 0);
        c.setValue(2, 0, 0, 0);
        c.setValue(2, 1, 0, -1);
        c.setValue(2, 2, 0, 1);
        c.setValue(2, 3, 0, 0);
        c.setValue(3, 0, 0, 0);
        c.setValue(3, 1, 0, 1);
        c.setValue(3, 2, 0, 0);
        c.setValue(3, 3, 0, -1);
        maskZeros(c);
        Matrix CT = c.transpose();

        G = new DMatrix(4, 3, 1);
        G.setValue(0, 0, 0, 1);
        G.setValue(0, 1, 0, 0);
        G.setValue(0, 2, 0, 0);
        G.setValue(1, 0, 0, 1/(double)2);
        G.setValue(1, 1, 0, 1/(double)2);
        G.setValue(1, 2, 0, 1/(double)2);
        G.setValue(2, 0, 0, 1/(double)2);
        G.setValue(2, 1, 0, -1/(double)2);
        G.setValue(2, 2, 0, 1/(double)2);
        G.setValue(3, 0, 0, 0);
        G.setValue(3, 1, 0, 0);
        G.setValue(3, 2, 0, 1);
        maskZeros(G);
        GT = G.transpose();

        winogradConvolutionMatrixOperation = new WinogradConvolutionMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), a, AT, c, CT);
        crosscorrelationInputGradientMatrixOperation = new CrosscorrelationInputGradientMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), argument1.getDepth(), argument2.getRows(), argument2.getColumns(), dilation, stride, false);
        crosscorrelationFilterGradientMatrixOperation = new CrosscorrelationFilterGradientMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), argument1.getDepth(), argument2.getRows(), argument2.getColumns(), dilation, stride, false);
    }

    /**
     * Masks matrix positions with zero value to avoid unnecessary calculations.
     *
     * @param matrix matrix to be masked.
     */
    private void maskZeros(Matrix matrix) {
        matrix.setMask();
        int rows = matrix.getRows();
        int columns = matrix.getColumns();
        int totalDepth = matrix.getDepth();
        for (int depth = 0; depth < totalDepth; depth++) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    if (matrix.getValue(row, column, depth) == 0) matrix.getMask().setMask(row, column, depth, true);
                }
            }
        }
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
