/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.matrix.operation.MaxPoolGradientMatrixOperation;
import utils.matrix.operation.MaxPoolMatrixOperation;
import utils.procedure.node.Node;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Implements expression for max pooling operation.<br>
 *
 */
public class MaxPoolExpression extends AbstractUnaryExpression implements Serializable {

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
        maxPoolMatrixOperation = new MaxPoolMatrixOperation(result.getRows(), result.getColumns(), argument1.getColumns(), filterRowSize, filterColumnSize, stride);
        maxPoolGradientMatrixOperation = new MaxPoolGradientMatrixOperation(result.getRows(), result.getColumns(), argument1.getColumns(), stride);
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
        if (argument1.getMatrix(sampleIndex) == null) throw new MatrixException(getExpressionName() + ": Arguments for operation not defined");
        if (maxPos == null) maxPos = new HashMap<>();
        maxPos.put(sampleIndex, new HashMap<>());
        maxPoolMatrixOperation.apply(argument1.getMatrix(sampleIndex), maxPos.get(sampleIndex), result.getNewMatrix(sampleIndex));
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
        if (!maxPos.containsKey(sampleIndex)) throw new MatrixException("Maximum positions for gradient calculation are not defined.");
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, maxPoolGradientMatrixOperation.apply(result.getGradient(sampleIndex), maxPos.get(sampleIndex), argument1.getEmptyMatrix()), false);
        maxPos.remove(sampleIndex);
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
