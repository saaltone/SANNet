/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.procedure.node.Node;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Class that describes expression for max pooling operation.<br>
 *
 */
public class MaxPoolExpression extends AbstractUnaryExpression implements Serializable {

    /**
     * Stride of max pooling operation.
     *
     */
    private final int stride;

    /**
     * Row size of filter.
     *
     */
    private final int filterRowSize;

    /**
     * Column size of filter.
     *
     */
    private final int filterColumnSize;

    /**
     * Maximum positions for max pool operation.
     *
     */
    private final HashMap<Integer, HashMap<Integer, Integer>> maxPos = new HashMap<>();

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
        this.stride = stride;
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
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
        argument1.getMatrix(index).setStride(stride);
        argument1.getMatrix(index).setFilterRowSize(filterRowSize);
        argument1.getMatrix(index).setFilterColumnSize(filterColumnSize);
        maxPos.put(index, new HashMap<>());
        result.setMatrix(index, argument1.getMatrix(index).maxPool(maxPos.get(index)));
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
        result.getGradient(index).setFilterRowSize(filterRowSize);
        result.getGradient(index).setFilterColumnSize(filterColumnSize);
        argument1.cumulateGradient(index, result.getGradient(index).maxPoolGradient(maxPos.remove(index)), false);
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
