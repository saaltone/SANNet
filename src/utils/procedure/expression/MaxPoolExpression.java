/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.operation.MaxPoolGradientMatrixOperation;
import utils.matrix.operation.MaxPoolMatrixOperation;
import utils.procedure.node.Node;

import java.util.HashMap;

/**
 * Implements expression for max pooling operation.<br>
 *
 */
public class MaxPoolExpression extends AbstractUnaryExpression {

    /**
     * Reference to max pool matrix operation.
     *
     */
    private final MaxPoolMatrixOperation maxPoolMatrixOperation;

    /**
     * Reference to max pool gradient matrix operation.
     *
     */
    private final MaxPoolGradientMatrixOperation maxPoolGradientMatrixOperation;

    /**
     * Maximum positions for max pool operation.
     *
     */
    private transient HashMap<Integer, HashMap<Integer, Integer>> maxPos = new HashMap<>();

    /**
     * Constructor for max pooling operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @param dilation dilation of pooling operation.
     * @param stride stride of pooling operation.
     * @param filterRowSize filter row size for operation.
     * @param filterColumnSize filter column size for operation.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public MaxPoolExpression(int expressionID, Node argument1, Node result, int dilation, int stride, int filterRowSize, int filterColumnSize) throws MatrixException {
        super("MAX_POOL", "MAX_POOL", expressionID, argument1, result);

        maxPoolMatrixOperation = new MaxPoolMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), filterRowSize, filterColumnSize, dilation, stride);
        maxPoolGradientMatrixOperation = new MaxPoolGradientMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), argument1.getRows(), argument1.getColumns(), stride);
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
        maxPos = new HashMap<>();
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
        maxPos.put(sampleIndex, new HashMap<>());
        return maxPoolMatrixOperation.apply(argument1.getMatrix(sampleIndex), maxPos.get(sampleIndex));
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
        HashMap<Integer, Integer> maxPosEntry = maxPos.get(sampleIndex);
        if (maxPosEntry == null) throw new MatrixException("Maximum positions for gradient calculation are not defined.");
        return maxPoolGradientMatrixOperation.apply(result.getGradient(sampleIndex), maxPosEntry);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        print();
        System.out.println(getExpressionName() + "(" + argument1.getName() + ") = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(false, getExpressionName() + "_GRADIENT(" + getResultGradientName() + ", ARGMAX(" + argument1.getName() +"))");
    }

}
