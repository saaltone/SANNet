/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.matrix.operation.*;
import utils.procedure.node.Node;

/**
 * Implements expression for convolution operation.<br>
 *
 */
public class ConvolveExpression extends AbstractBinaryExpression {

    /**
     * Reference to convolution matrix operation.
     *
     */
    private final ConvolutionMatrixOperation convolutionMatrixOperation;

    /**
     * Reference to convolution input gradient matrix operation.
     *
     */
    private final ConvolutionInputGradientMatrixOperation convolutionInputGradientMatrixOperation;

    /**
     * Reference to convolution filter gradient matrix operation.
     *
     */
    private final ConvolutionFilterGradientMatrixOperation convolutionFilterGradientMatrixOperation;

    /**
     * Constructor for convolution operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of expression.
     * @param stride stride of convolution operation.
     * @param dilation dilation step size for convolution operation.
     * @param isDepthSeparable if true convolution is depth separable
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public ConvolveExpression(int expressionID, Node argument1, Node argument2, Node result, int stride, int dilation, boolean isDepthSeparable) throws MatrixException {
        super("CONVOLVE", "CONVOLVE", expressionID, argument1, argument2, result);

        convolutionMatrixOperation = new ConvolutionMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), result.getRows(), argument2.getColumns(), dilation, stride, isDepthSeparable);
        convolutionInputGradientMatrixOperation = new ConvolutionInputGradientMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), argument2.getRows(), argument2.getColumns(), dilation, stride, isDepthSeparable);
        convolutionFilterGradientMatrixOperation = new ConvolutionFilterGradientMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), argument2.getRows(), argument2.getColumns(), dilation, stride, isDepthSeparable);
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
        checkArguments(argument1, argument2, sampleIndex);
        result.setMatrix(sampleIndex, convolutionMatrixOperation.apply(argument1.getMatrix(sampleIndex), argument2.getMatrix(sampleIndex)));
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
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, convolutionInputGradientMatrixOperation.apply(result.getGradient(sampleIndex), argument2.getMatrix(sampleIndex)), false);
        if (!argument2.isStopGradient()) argument2.cumulateGradient(sampleIndex, convolutionFilterGradientMatrixOperation.apply(result.getGradient(sampleIndex), argument1.getMatrix(sampleIndex)), false);
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
