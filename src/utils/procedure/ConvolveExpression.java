/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import utils.matrix.MatrixException;

import java.io.Serializable;

/**
 * Class that describes expression for convolution operation.
 *
 */
public class ConvolveExpression extends AbstractBinaryExpression implements Serializable {

    /**
     * Stride of convolution operation.
     *
     */
    private final int stride;

    /**
     * Dilation step size for cross-correlation operation.
     *
     */
    private final int dilation;

    /**
     * Filter size;
     *
     */
    private final int filterSize;

    /**
     * Constructor for convolution operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of expression.
     * @param stride stride of convolution operation.
     * @param dilation dilation step size for cross-correlation operation.
     * @param filterSize filter size.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public ConvolveExpression(int expressionID, Node argument1, Node argument2, Node result, int stride, int dilation, int filterSize) throws MatrixException {
        super(expressionID, argument1, argument2, result);
        this.stride = stride;
        this.dilation = dilation;
        this.filterSize = filterSize;
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
        if (argument1.getMatrix(index) == null || argument2.getMatrix(index) == null) throw new MatrixException("Arguments for CONVOLVE operation not defined");
        argument1.getMatrix(index).setStride(stride);
        argument1.getMatrix(index).setDilation(dilation);
        argument1.getMatrix(index).setFilterSize(filterSize);
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
        if (result.getGradient(index) == null) throw new MatrixException("Result gradient not defined.");
        result.getGradient(index).setStride(stride);
        result.getGradient(index).setDilation(dilation);
        result.getGradient(index).setFilterSize(filterSize);
        argument1.updateGradient(index, result.getGradient(index).convolveOutputGradient(argument2.getMatrix(index)), true);
        argument2.updateGradient(index, result.getGradient(index).convolveFilterGradient(argument1.getMatrix(index)), true);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        System.out.print("Expression " +getExpressionID() + ": ");
        System.out.println("CONVOLVE(" + argument1.getName() + ", " + argument2.getName() + ") = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        System.out.print("Expression " +getExpressionID() + ": ");
        System.out.println("CONVOLVE: d" + argument1.getName() + " = CONVOLVE_GRADIENT(d" + result.getName() + ", " + argument2.getName() + ")");
        System.out.print("Expression " +getExpressionID() + ": ");
        System.out.println("CONVOLVE: d" + argument2.getName() + " = CONVOLVE_GRADIENT(d" + result.getName() + ", " + argument1.getName() + ")");
    }

}
