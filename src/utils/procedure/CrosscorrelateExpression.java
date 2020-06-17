/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import utils.matrix.MatrixException;

import java.io.Serializable;

/**
 * Class that describes expression for cross-correlation operation.
 *
 */
public class CrosscorrelateExpression extends AbstractBinaryExpression implements Serializable {

    /**
     * Stride of cross-correlation operation.
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
     * Constructor for cross-correlation operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of expression.
     * @param stride stride of cross-correlation operation.
     * @param dilation dilation step size for cross-correlation operation.
     * @param filterSize filter size.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public CrosscorrelateExpression(int expressionID, Node argument1, Node argument2, Node result, int stride, int dilation, int filterSize) throws MatrixException {
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
        if (argument1.getMatrix(index) == null || argument2.getMatrix(index) == null) throw new MatrixException("Arguments for CROSSCORRELATE operation not defined");
        argument1.getMatrix(index).setStride(stride);
        argument1.getMatrix(index).setDilation(dilation);
        argument1.getMatrix(index).setFilterSize(filterSize);
        result.setMatrix(index, argument1.getMatrix(index).crosscorrelate(argument2.getMatrix(index)));
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
        argument1.updateGradient(index, result.getGradient(index).crosscorrelateOutputGradient(argument2.getMatrix(index)), true);
        argument2.updateGradient(index, result.getGradient(index).crosscorrelateFilterGradient(argument1.getMatrix(index)), true);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        System.out.print("Expression " +getExpressionID() + ": ");
        System.out.println("CROSSCORRELATE(" + argument1.getName() + ", " + argument2.getName() + ") = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        System.out.print("Expression " +getExpressionID() + ": ");
        System.out.println("CROSSCORRELATE: d" + argument1.getName() + " = CROSSCORRELATE_GRADIENT(d" + result.getName() + ", " + argument2.getName() + ")");
        System.out.print("Expression " +getExpressionID() + ": ");
        System.out.println("CROSSCORRELATE: d" + argument2.getName() + " = CROSSCORRELATE_GRADIENT(d" + result.getName() + ", " + argument1.getName() + ")");
    }

}
