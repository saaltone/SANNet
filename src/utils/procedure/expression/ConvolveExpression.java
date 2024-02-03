/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.Matrix;
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
     * @param expressionID     unique ID for expression.
     * @param argument1        first argument.
     * @param argument2        second argument.
     * @param result           result of expression.
     * @param stride           stride of convolution operation.
     * @param dilation         dilation step size for convolution operation.
     * @param isDepthSeparable if true convolution is depth separable
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public ConvolveExpression(int expressionID, Node argument1, Node argument2, Node result, int stride, int dilation, boolean isDepthSeparable) throws MatrixException {
        super("CONVOLVE", expressionID, argument1, argument2, result);

        convolutionMatrixOperation = new ConvolutionMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), argument1.getDepth(), argument2.getRows(), argument2.getColumns(), dilation, stride, isDepthSeparable);
        convolutionInputGradientMatrixOperation = new ConvolutionInputGradientMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), argument1.getDepth(), argument2.getRows(), argument2.getColumns(), dilation, stride, isDepthSeparable);
        convolutionFilterGradientMatrixOperation = new ConvolutionFilterGradientMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), argument1.getDepth(), argument2.getRows(), argument2.getColumns(), dilation, stride, isDepthSeparable);
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
        return convolutionMatrixOperation.apply(argument1Matrix, argument2Matrix);
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
        return convolutionInputGradientMatrixOperation.apply(resultGradient, argument2Matrix);
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
        return convolutionFilterGradientMatrixOperation.apply(resultGradient, argument1Matrix);
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
