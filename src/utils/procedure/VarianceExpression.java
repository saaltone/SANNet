/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.procedure;

import utils.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;

/**
 * Class that describes expression for variance operation.
 *
 */
public class VarianceExpression extends AbstractUnaryExpression implements Serializable {

    /**
     * Name of operation.
     *
     */
    private static final String operationName = "VARIANCE";

    /**
     * True if calculation is done as multi matrix.
     *
     */
    private final boolean asMultiMatrix;

    /**
     * Mean value as matrix.
     *
     */
    private Matrix mean;

    /**
     * Constructor for variance operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @param asMultiMatrix true if calculation is done per index otherwise over all indices.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public VarianceExpression(int expressionID, Node argument1, Node result, boolean asMultiMatrix) throws MatrixException {
        super(operationName, operationName, expressionID, argument1, result);
        this.asMultiMatrix = asMultiMatrix;
    }

    /**
     * Calculates expression.
     *
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateExpression() throws MatrixException, DynamicParamException {
        if (!asMultiMatrix) return;
        if (argument1.getMatrices() == null) throw new MatrixException("Arguments for VARIANCE operation not defined");
        mean = argument1.getMatrices().mean();
        result.setMultiIndex(false);
        result.setMatrix(argument1.getMatrices().variance(mean));
    }

    /**
     * Calculates expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int index) throws MatrixException {
        if (asMultiMatrix) return;
        if (argument1.getMatrix(index) == null) throw new MatrixException("Arguments for VARIANCE operation not defined");
        mean = argument1.getMatrix(index).meanAsMatrix();
        result.setMatrix(index, argument1.getMatrix(index).varianceAsMatrix(mean));
    }

    /**
     * Calculates gradient of expression.
     *
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateGradient() throws MatrixException {
        if (!asMultiMatrix) return;
        if (result.getGradient() == null) throw new MatrixException("Result gradient not defined.");
        for (Integer index : argument1.keySet()) {
            Matrix varianceGradient = argument1.getMatrix(index).subtract(mean).multiply(2 / (double)argument1.size());
            argument1.updateGradient(index, result.getGradient().multiply(varianceGradient),true);
        }
    }

    /**
     * Calculates gradient of expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int index) throws MatrixException {
        if (asMultiMatrix) return;
        if (result.getGradient(index) == null) throw new MatrixException("Result gradient not defined.");
        Matrix varianceGradient = argument1.getMatrix(index).subtract(mean).multiply(2 / (double)result.getGradient(index).size());
        argument1.updateGradient(index, result.getGradient(index).multiply(varianceGradient), true);
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
        printArgument1Gradient(true, " * (" + argument1.getName() + " - MEAN("  + argument1.getName() + ")) * 2 / SIZE(" + argument1.getName() + ")");
    }

}
