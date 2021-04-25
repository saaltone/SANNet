/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Class that describes expression for average pooling operation.<br>
 *
 */
public class AveragePoolExpression extends AbstractUnaryExpression implements Serializable {

    /**
     * Stride of average pooling operation.
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
     * Constructor for average pool expression.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @param stride stride of pooling operation.
     * @param filterRowSize filter row size for operation.
     * @param filterColumnSize filter column size for operation.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public AveragePoolExpression(int expressionID, Node argument1, Node result, int stride, int filterRowSize, int filterColumnSize) throws MatrixException {
        super("AVERAGE_POOL", "AVERAGE_POOL", expressionID, argument1, result);
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
        if (result.getGradient(index) == null) throw new MatrixException(getExpressionName() + ": Result gradient not defined.");
        result.getGradient(index).setStride(stride);
        result.getGradient(index).setFilterRowSize(filterRowSize);
        result.getGradient(index).setFilterColumnSize(filterColumnSize);
        argument1.cumulateGradient(index, result.getGradient(index).averagePoolGradient(), false);
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
        print();
        System.out.println(getArgument1PrefixName() + "_GRADIENT(" + result.getName() + ")" + getArgument1SumPostfix());
    }

}
