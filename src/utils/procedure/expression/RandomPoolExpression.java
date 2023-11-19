/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.operation.RandomPoolGradientMatrixOperation;
import utils.matrix.operation.RandomPoolMatrixOperation;
import utils.procedure.node.Node;

import java.util.HashMap;

/**
 * Implements expression for random pooling operation.<br>
 *
 */
public class RandomPoolExpression extends AbstractUnaryExpression {

    /**
     * Reference to random pool matrix operation.
     *
     */
    private final RandomPoolMatrixOperation randomPoolMatrixOperation;

    /**
     * Reference to random pool gradient matrix operation.
     *
     */
    private final RandomPoolGradientMatrixOperation randomPoolGradientMatrixOperation;

    /**
     * Input positions for random pool operation.
     *
     */
    private transient HashMap<Integer, HashMap<Integer, Integer>> inputPos = new HashMap<>();

    /**
     * Constructor for random pooling operation.
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
    public RandomPoolExpression(int expressionID, Node argument1, Node result, int dilation, int stride, int filterRowSize, int filterColumnSize) throws MatrixException {
        super("RANDOM_POOL", "RANDOM_POOL", expressionID, argument1, result);

        randomPoolMatrixOperation = new RandomPoolMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), filterRowSize, filterColumnSize, dilation, stride);
        randomPoolGradientMatrixOperation = new RandomPoolGradientMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), argument1.getRows(), argument1.getColumns(), stride);
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
        inputPos = new HashMap<>();
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
        inputPos.put(sampleIndex, new HashMap<>());
        return randomPoolMatrixOperation.apply(argument1Matrix, inputPos.get(sampleIndex));
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
        HashMap<Integer, Integer> inputPosEntry = inputPos.get(sampleIndex);
        if (inputPosEntry == null) throw new MatrixException("Input positions for gradient calculation are not defined.");
        return randomPoolGradientMatrixOperation.apply(result.getGradient(sampleIndex), inputPosEntry);
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
