/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

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
     * @param stride stride of pooling operation.
     * @param filterRowSize filter row size for operation.
     * @param filterColumnSize filter column size for operation.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public MaxPoolExpression(int expressionID, Node argument1, Node result, int stride, int filterRowSize, int filterColumnSize) throws MatrixException {
        super("MAX_POOL", "MAX_POOL", expressionID, argument1, result);

        maxPoolMatrixOperation = new MaxPoolMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), filterRowSize, filterColumnSize, stride);
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
        checkArgument(argument1, sampleIndex);
        HashMap<Integer, Integer> maxPosEntry = new HashMap<>();
        maxPos.put(sampleIndex, maxPosEntry);
        result.setMatrix(sampleIndex, maxPoolMatrixOperation.apply(argument1.getMatrix(sampleIndex), maxPosEntry));
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
        HashMap<Integer, Integer> maxPosEntry = maxPos.get(sampleIndex);
        if (maxPosEntry == null) throw new MatrixException("Maximum positions for gradient calculation are not defined.");
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, maxPoolGradientMatrixOperation.apply(result.getGradient(sampleIndex), maxPosEntry), false);
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
