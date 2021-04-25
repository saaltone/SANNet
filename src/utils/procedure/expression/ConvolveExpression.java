/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Class that describes expression for convolution operation.<br>
 *
 */
public class ConvolveExpression extends AbstractBinaryExpression implements Serializable {

    /**
     * Stride of convolution operation.
     *
     */
    private final int stride;

    /**
     * Dilation step size for convolution operation.
     *
     */
    private final int dilation;

    /**
     * Filter row size;
     *
     */
    private final int filterRowSize;

    /**
     * Filter column size;
     *
     */
    private final int filterColumnSize;

    /**
     * Constructor for convolution operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of expression.
     * @param stride stride of convolution operation.
     * @param dilation dilation step size for convolution operation.
     * @param filterRowSize filter row size.
     * @param filterColumnSize filter column size.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public ConvolveExpression(int expressionID, Node argument1, Node argument2, Node result, int stride, int dilation, int filterRowSize, int filterColumnSize) throws MatrixException {
        super("CONVOLVE", "CONVOLVE", expressionID, argument1, argument2, result);
        this.stride = stride;
        this.dilation = dilation;
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
        if (argument1.getMatrix(index) == null || argument2.getMatrix(index) == null) throw new MatrixException(getExpressionName() + "Arguments for operation not defined");
        argument1.getMatrix(index).setStride(stride);
        argument1.getMatrix(index).setDilation(dilation);
        argument1.getMatrix(index).setFilterRowSize(filterRowSize);
        argument1.getMatrix(index).setFilterColumnSize(filterColumnSize);
        result.setMatrix(index, argument1.getMatrix(index).convolve(argument2.getMatrix(index)));
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
        result.getGradient(index).setDilation(dilation);
        result.getGradient(index).setFilterRowSize(filterRowSize);
        result.getGradient(index).setFilterColumnSize(filterColumnSize);
        argument1.cumulateGradient(index, result.getGradient(index).convolveInputGradient(argument2.getMatrix(index)), false);
        argument2.cumulateGradient(index, result.getGradient(index).convolveFilterGradient(argument1.getMatrix(index)), false);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        printSpecificBinaryExpression();
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(false, getExpressionName() + "_GRADIENT(d" + result.getName() + ", " + argument2.getName() + ")");
        printArgument2Gradient(false, false, getExpressionName() + "_GRADIENT(d" + result.getName() + ", " + argument1.getName() + ")");
    }

}
