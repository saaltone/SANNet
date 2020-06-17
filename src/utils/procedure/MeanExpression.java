/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;

/**
 * Class that describes expression for mean function.
 *
 */
public class MeanExpression extends AbstractUnaryExpression implements Serializable {

    /**
     * True if calculation is done as multi matrix.
     *
     */
    private final boolean asMultiMatrix;

    /**
     * Constructor for mean operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @param asMultiMatrix true if calculation is done per index otherwise over all indices.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public MeanExpression(int expressionID, Node argument1, Node result, boolean asMultiMatrix) throws MatrixException {
        super(expressionID, argument1, result);
        this.asMultiMatrix = asMultiMatrix;
    }

    /**
     * Calculates expression.
     *
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression() throws MatrixException {
        if (!asMultiMatrix) return;
        if (argument1.getMatrices() == null) throw new MatrixException("Arguments for MEAN operation not defined");
        Matrix mean = argument1.getMatrices().mean();
        result.setMultiIndex(false);
        result.setMatrix(mean);
    }

    /**
     * Calculates expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int index) throws MatrixException {
        if (asMultiMatrix) return;
        if (argument1.getMatrix(index) == null) throw new MatrixException("Arguments for MEAN operation not defined");
        result.setMatrix(index, argument1.getMatrix(index).meanAsMatrix());
    }

    /**
     * Calculates gradient of expression.
     *
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateGradient() throws MatrixException {
        if (!asMultiMatrix) return;
        if (result.getGradient() == null) throw new MatrixException("Result gradient not defined.");
        Matrix meanGradient = result.getGradient().multiply(1 / (double)argument1.size());
        for (Integer index : argument1.keySet()) argument1.updateGradient(index, meanGradient, true);
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
        Matrix meanGradient = result.getGradient(index).multiply(1 / (double)result.getGradient(index).size());
        argument1.updateGradient(index, meanGradient, true);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        System.out.print("Expression " +getExpressionID() + ": ");
        System.out.println("MEAN(" + argument1.getName() + ") = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        System.out.print("Expression " +getExpressionID() + ": ");
        System.out.println("MEAN: d" + argument1.getName() + " = d" + result.getName() + " / SIZE(" + argument1.getName() + ")");
    }

}
