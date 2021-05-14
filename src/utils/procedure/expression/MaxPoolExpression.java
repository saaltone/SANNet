/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.matrix.operation.MaxPoolGradientMatrixOperation;
import utils.matrix.operation.MaxPoolMatrixOperation;
import utils.procedure.node.Node;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Stack;

/**
 * Class that describes expression for max pooling operation.<br>
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
    private final HashMap<Integer, HashMap<Integer, Integer>> maxPos = new HashMap<>();

    /**
     * Stack for caching maximum position instances.
     *
     */
    private final Stack<HashMap<Integer, Integer>> maxPosCache = new Stack<>();

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
     * Calculates expression.
     *
     */
    public void calculateExpression() {
    }

    /**
     * Calculates expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int index) throws MatrixException {
        if (argument1.getMatrix(index) == null) throw new MatrixException(getExpressionName() + ": Arguments for operation not defined");
        if (!maxPosCache.empty()) maxPos.put(index, maxPosCache.pop());
        else maxPos.put(index, new HashMap<>());
        maxPoolMatrixOperation.apply(argument1.getMatrix(index), maxPos.get(index), result.getNewMatrix(index));
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
     * @param index data index.
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int index) throws MatrixException {
        if (result.getGradient(index) == null) throw new MatrixException(getExpressionName() + ": Result gradient not defined.");
        if (!maxPos.containsKey(index)) throw new MatrixException("Maximum positions for gradient calculation are not defined.");
        argument1.cumulateGradient(index, maxPoolGradientMatrixOperation.apply(result.getGradient(index), maxPos.get(index), argument1.getEmptyMatrix()), false);
        maxPosCache.push(maxPos.remove(index));
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
