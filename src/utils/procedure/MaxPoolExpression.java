/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import utils.matrix.MatrixException;

import java.io.Serializable;

/**
 * Class that describes expression for max pooling operation.
 *
 */
public class MaxPoolExpression extends AbstractUnaryExpression implements Serializable {

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
     * @param arg1 first argument.
     * @param result result of expression.
     * @param stride stride of pooling operation.
     * @param poolSize pool size of pooling operation.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public MaxPoolExpression(int expressionID, Node arg1, Node result, int stride, int poolSize) throws MatrixException {
        super(expressionID, arg1, result);
        this.stride = stride;
        this.poolSize = poolSize;
    }

    /**
     * Calculates expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int index) throws MatrixException {
        if (arg1.getMatrix(index) == null) throw new MatrixException("Arguments for MAXPOOL operation not defined");
        arg1.getMatrix(index).setStride(stride);
        arg1.getMatrix(index).setPoolSize(poolSize);
        maxArgsAt = new int[arg1.getMatrix(index).getRows() - poolSize + 1][arg1.getMatrix(index).getCols() - poolSize + 1][2];
        result.setMatrix(index, arg1.getMatrix(index).maxPool(maxArgsAt));
    }

    /**
     * Calculates gradient of expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int index) throws MatrixException {
        if (result.getGradient(index) == null) throw new MatrixException("Result gradient not defined.");
        if (maxArgsAt == null) throw new MatrixException("Maximum arguments for gradient calculation are not defined.");
        result.getGradient(index).setStride(stride);
        result.getGradient(index).setPoolSize(poolSize);
        arg1.updateGradient(index, result.getGradient(index).maxPoolGrad(maxArgsAt), true);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        System.out.print("MAXPOOL: " + arg1 + " " + result);
    }

}
