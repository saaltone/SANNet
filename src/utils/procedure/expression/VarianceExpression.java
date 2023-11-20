/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.configurable.DynamicParamException;
import utils.matrix.AbstractMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.procedure.node.Node;

import java.util.HashMap;
import java.util.Map;

/**
 * Implements expression for variance operation.<br>
 *
 */
public class VarianceExpression extends AbstractUnaryExpression {

    /**
     * True if calculation is done as single step otherwise false.
     *
     */
    private final boolean executeAsSingleStep;

    /**
     * Mean value as matrix for multi matrix case.
     *
     */
    private transient Matrix mean;

    /**
     * Mean values as matrix for non-multi matrix case.
     *
     */
    private transient HashMap<Integer, Matrix> means = new HashMap<>();

    /**
     * Constructor for variance operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @param executeAsSingleStep true if calculation is done per index otherwise over all indices.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public VarianceExpression(int expressionID, Node argument1, Node result, boolean executeAsSingleStep) throws MatrixException {
        super("VARIANCE", expressionID, argument1, result);

        this.executeAsSingleStep = executeAsSingleStep;
    }

    /**
     * Returns true is expression is executed as single step otherwise false.
     *
     * @return true is expression is executed as single step otherwise false.
     */
    protected boolean executeAsSingleStep() {
        return executeAsSingleStep;
    }

    /**
     * Resets expression.
     *
     */
    public void applyReset() {
        mean = null;
        means = new HashMap<>();
    }

    /**
     * Calculates result matrix.
     *
     * @return result matrix.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected Matrix calculateResult() throws MatrixException, DynamicParamException {
        mean = AbstractMatrix.mean(argument1.getMatrices());
        return AbstractMatrix.variance(argument1.getMatrices(), mean);
    }

    /**
     * Calculates result matrix.
     *
     * @param sampleIndex sample index
     * @param argument1Matrix argument1 matrix for a sample index.
     * @param argument2Matrix argument2 matrix for a sample index.
     * @return result matrix.
     * @throws MatrixException throws exception if calculation fails.
     */
    protected Matrix calculateResult(int sampleIndex, Matrix argument1Matrix, Matrix argument2Matrix) throws MatrixException {
        Matrix mean = argument1Matrix.meanAsMatrix();
        if (means == null) means = new HashMap<>();
        means.put(sampleIndex, mean);
        return argument1Matrix.varianceAsMatrix(mean);
    }

    /**
     * Calculates argument 1 gradient matrix.
     *
     * @throws MatrixException throws exception if calculation fails.
     */
    protected void calculateArgument1Gradient() throws MatrixException {
        double argument1Size = argument1.size();
        for (Map.Entry<Integer, Matrix> entry : argument1.entrySet()) {
            argument1.cumulateGradient(entry.getKey(), result.getGradient().multiply(entry.getValue().subtract(mean).multiply(2 / argument1Size)));
        }
    }

    /**
     * Calculates argument 1 gradient matrix.
     *
     * @param sampleIndex     sample index.
     * @param resultGradient  result gradient.
     * @param argument1Matrix argument 1 matrix.
     * @param argument2Matrix argument 2 matrix.
     * @param resultMatrix    result matrix.
     * @return argument1 gradient matrix.
     * @throws MatrixException throws exception if calculation fails.
     */
    protected Matrix calculateArgument1Gradient(int sampleIndex, Matrix resultGradient, Matrix argument1Matrix, Matrix argument2Matrix, Matrix resultMatrix) throws MatrixException {
        return resultGradient.multiply(argument1Matrix.subtract(means.get(sampleIndex)).multiply(2 / (double)argument1Matrix.size()));
    }

    /**
     * Returns expression operation signature.
     *
     * @return expression operation signature.
     */
    protected String getExpressionOperationSignature() {
        return getExpressionName() + "(" + getArgument1().getName() + ")";
    }

    /**
     * Returns gradient 1 operation signature.
     *
     * @return gradient 1 operation signature.
     */
    protected String getGradientOperation1Signature() {
        return "d" + getResult().getName() + " * (" + getArgument1().getName() + " - MEAN(" + getArgument1().getName() + ")) * 2 / SIZE(" + getArgument1().getName() + ")";
    }

}
