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
     * Constructor for cross-correlation operation.
     *
     * @param expressionID unique ID for expression.
     * @param arg1 first argument.
     * @param arg2 second argument.
     * @param result result of expression.
     * @param stride stride of cross-correlation operation.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public CrosscorrelateExpression(int expressionID, Node arg1, Node arg2, Node result, int stride) throws MatrixException {
        super(expressionID, arg1, arg2, result);
        this.stride = stride;
    }

    /**
     * Calculates expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int index) throws MatrixException {
        if (arg1.getMatrix(index) == null || arg2.getMatrix(index) == null) throw new MatrixException("Arguments for CROSSCORRELATE operation not defined");
        arg1.getMatrix(index).setStride(stride);
        result.setMatrix(index, arg1.getMatrix(index).crosscorrelate(arg2.getMatrix(index)));
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
        arg1.updateGradient(index, result.getGradient(index).crosscorrelateOutGrad(arg2.getMatrix(index)), true);
        arg2.updateGradient(index, result.getGradient(index).crosscorrelateFilterGrad(arg1.getMatrix(index)), true);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        System.out.print("CROSSCORRELATE: " + arg1 + " " + arg2 + " " + result);
    }

}
