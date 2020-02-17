/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import utils.matrix.MatrixException;

import java.io.Serializable;

/**
 * Class that describes expression for average pooling operation.
 *
 */
public class AveragePoolExpression extends AbstractUnaryExpression implements Serializable {

    /**
     * Stride of average pooling operation.
     *
     */
    private final int stride;

    /**
     * Size of pool.
     *
     */
    private final int poolSize;

    /**
     * Constructor for average pool expression.
     *
     * @param expressionID unique ID for expression.
     * @param arg1 first argument.
     * @param result result of expression.
     * @param stride stride of pooling operation.
     * @param poolSize pool size of pooling operation.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public AveragePoolExpression(int expressionID, Node arg1, Node result, int stride, int poolSize) throws MatrixException {
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
        if (arg1.getMatrix(index) == null) throw new MatrixException("Arguments for AVERAGE POOL operation not defined");
        arg1.getMatrix(index).setStride(stride);
        arg1.getMatrix(index).setPoolSize(poolSize);
        result.setMatrix(index, arg1.getMatrix(index).avgPool());
    }

    /**
     * Calculates gradient of expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int index) throws MatrixException {
        if (result.getGradient(index) == null) throw new MatrixException("Result gradient not defined.");
        result.getGradient(index).setStride(stride);
        result.getGradient(index).setPoolSize(poolSize);
        arg1.updateGradient(index, result.getGradient(index).avgPoolGrad(), true);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        System.out.print("AVERAGE POOL: " + arg1 + " " + result);
    }

}
