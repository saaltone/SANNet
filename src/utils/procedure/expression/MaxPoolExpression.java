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
     * Name of operation.
     *
     */
    private static final String expressionName = "MAX POOL";

    /**
     * Stride of max pooling operation.
     *
     */
    private final int stride;

    /**
     * Row size of pool.
     *
     */
    private final int poolRowSize;

    /**
     * Column size of pool.
     *
     */
    private final int poolColumnSize;

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
     * @param poolRowSize pool row size for operation.
     * @param poolColumnSize pool column size for operation.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public MaxPoolExpression(int expressionID, Node argument1, Node result, int stride, int poolRowSize, int poolColumnSize) throws MatrixException {
        super(expressionName, expressionName, expressionID, argument1, result);
        this.stride = stride;
        this.poolRowSize = poolRowSize;
        this.poolColumnSize = poolColumnSize;
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
        if (argument1.getMatrix(index) == null) throw new MatrixException(expressionName + ": Arguments for operation not defined");
        argument1.getMatrix(index).setStride(stride);
        argument1.getMatrix(index).setPoolRowSize(poolRowSize);
        argument1.getMatrix(index).setPoolColumnSize(poolColumnSize);
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
        if (result.getGradient(index) == null) throw new MatrixException(expressionName + ": Result gradient not defined.");
        if (!maxPos.containsKey(index)) throw new MatrixException("Maximum positions for gradient calculation are not defined.");
        result.getGradient(index).setPoolRowSize(poolRowSize);
        result.getGradient(index).setPoolColumnSize(poolColumnSize);
        argument1.cumulateGradient(index, result.getGradient(index).maxPoolGradient(maxPos.remove(index)), false);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        print();
        System.out.println(getName() + "(" + argument1.getName() + ") = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(false, getName() + "_GRADIENT(" + getResultGradientName() + ", ARGMAX(" + argument1.getName() +"))");
    }

}
