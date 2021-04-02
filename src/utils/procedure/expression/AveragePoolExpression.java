/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Class that describes expression for average pooling operation.
 *
 */
public class AveragePoolExpression extends AbstractUnaryExpression implements Serializable {

    /**
     * Name of operation.
     *
     */
    private static final String expressionName = "AVERAGE POOL";

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
     * @param argument1 first argument.
     * @param result result of expression.
     * @param stride stride of pooling operation.
     * @param poolSize pool size of pooling operation.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public AveragePoolExpression(int expressionID, Node argument1, Node result, int stride, int poolSize) throws MatrixException {
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
        result.setMatrix(index, argument1.getMatrix(index).averagePool());
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
        result.getGradient(index).setStride(stride);
        result.getGradient(index).setPoolSize(poolSize);
        argument1.updateGradient(index, result.getGradient(index).averagePoolGradient(), true);
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
        print();
        System.out.println(getArgument1PrefixName() + "_GRADIENT(" + result.getName() + ")" + getArgument1SumPostfix());
    }

}
