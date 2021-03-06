/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunction;
import utils.matrix.UnaryFunctionType;
import utils.procedure.node.Node;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Class that describes expression for standard deviation operation.<br>
 *
 */
public class StandardDeviationExpression extends AbstractUnaryExpression implements Serializable {

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
     * Mean values as matrix for non-multi matrix case.
     *
     */
    private transient HashMap<Integer, Matrix> means = new HashMap<>();

    /**
     * Operation for square root.
     *
     */
    private final UnaryFunction sqrtFunction = new UnaryFunction(UnaryFunctionType.SQRT);

    /**
     * Constructor for standard deviation operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @param asMultiMatrix true if calculation is done per index otherwise over all indices.
     * @throws MatrixException throws exception if expression arguments are not defined.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public StandardDeviationExpression(int expressionID, Node argument1, Node result, boolean asMultiMatrix) throws MatrixException, DynamicParamException {
        super("STANDARD_DEVIATION", "STANDARD_DEVIATION", expressionID, argument1, result);
        this.asMultiMatrix = asMultiMatrix;
    }

    /**
     * Resets expression.
     *
     */
    public void reset() {
        mean = null;
        means = new HashMap<>();
        super.reset();
    }

    /**
     * Calculates expression.
     *
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateExpression() throws MatrixException, DynamicParamException {
        if (!asMultiMatrix) return;
        if (argument1.getMatrices() == null) throw new MatrixException(getExpressionName() + ": Arguments for operation not defined");
        mean = argument1.getMatrices().mean();
        Matrix standardDeviation = argument1.getMatrices().standardDeviation(mean);
        result.setMultiIndex(false);
        result.setMatrix(standardDeviation);
    }

    /**
     * Calculates expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int index) throws MatrixException {
        if (asMultiMatrix) return;
        if (argument1.getMatrix(index) == null) throw new MatrixException(getExpressionName() + "Arguments for operation not defined");
        Matrix mean = argument1.getMatrix(index).meanAsMatrix();
        if (means == null) means = new HashMap<>();
        means.put(index, mean);
        result.setMatrix(index, argument1.getMatrix(index).standardDeviationAsMatrix(mean));
    }

    /**
     * Calculates gradient of expression.
     *
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient() throws MatrixException {
        if (!asMultiMatrix) return;
        if (result.getGradient() == null) throw new MatrixException(getExpressionName() + ": Result gradient not defined.");
        for (Integer index : argument1.keySet()) {
            Matrix standardDeviationGradient = argument1.getMatrix(index).subtract(mean).multiply(2 / (double)argument1.size() - 1).apply(sqrtFunction.getDerivative());
            argument1.cumulateGradient(index, result.getGradient().multiply(standardDeviationGradient), false);
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
        if (result.getGradient(index) == null) throw new MatrixException(getExpressionName() + ": Result gradient not defined.");
        Matrix standardDeviationGradient = argument1.getMatrix(index).subtract(means.get(index)).multiply(2 / (double)(result.getGradient(index).size() - 1)).apply(sqrtFunction.getDerivative());
        argument1.cumulateGradient(index, result.getGradient(index).multiply(standardDeviationGradient), false);
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
        printArgument1Gradient(true, " * SQRT_GRADIENT((" + argument1.getName() + " - MEAN("  + argument1.getName() + ")) * 2 / SIZE(" + argument1.getName() + "))");
    }

}
