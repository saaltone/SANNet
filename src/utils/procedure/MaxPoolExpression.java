/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.procedure;

import utils.matrix.MatrixException;

import java.io.Serializable;

/**
 * Class that describes expression for max pooling operation.
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
     * Size of pool.
     *
     */
    private final int poolSize;

    /**
     * Maximum arguments for max pool operation.
     *
     */
    private int [][][] maxArgsAt;

    /**
     * Constructor for max pooling operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @param stride stride of pooling operation.
     * @param poolSize pool size of pooling operation.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public MaxPoolExpression(int expressionID, Node argument1, Node result, int stride, int poolSize) throws MatrixException {
        super(expressionName, expressionName, expressionID, argument1, result);
        this.stride = stride;
        this.poolSize = poolSize;
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
        argument1.getMatrix(index).setPoolSize(poolSize);
        maxArgsAt = new int[argument1.getMatrix(index).getRows() - poolSize + 1][argument1.getMatrix(index).getColumns() - poolSize + 1][2];
        result.setMatrix(index, argument1.getMatrix(index).maxPool(maxArgsAt));
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
        if (maxArgsAt == null) throw new MatrixException("Maximum arguments for gradient calculation are not defined.");
        result.getGradient(index).setStride(stride);
        result.getGradient(index).setPoolSize(poolSize);
        argument1.updateGradient(index, result.getGradient(index).maxPoolGradient(maxArgsAt), true);
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
        printArgument1Gradient(false, getName() + "_GRADIENT(" + getResultGradientName() + ", MAX_ARGS(" + argument1.getName() +"))");
    }

}
