/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.*;
import utils.matrix.operation.BinaryMatrixOperation;
import utils.matrix.operation.NormMatrixOperation;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Implements norm expression.<br>
 *
 */
public class NormExpression extends AbstractUnaryExpression implements Serializable {

    /**
     * Reference to norm matrix operation.
     *
     */
    private final NormMatrixOperation normMatrixOperation;

    /**
     * Reference to norm gradient matrix operation.
     *
     */
    private final BinaryMatrixOperation normGradientMatrixOperation;

    /**
     * Reference to multiply matrix operation.
     *
     */
    private final BinaryMatrixOperation multiplyMatrixOperation;

    /**
     * Power of norm.
     *
     */
    private final int p;

    /**
     * Constructor for norm operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @param p power of norm.
     * @throws MatrixException throws exception if expression arguments are not defined or norm p value is not at least 2.
     */
    public NormExpression(int expressionID, Node argument1, Node result, int p) throws MatrixException {
        super("NORM", "NORM", expressionID, argument1, result);

        if (p < 2) throw new MatrixException("Norm p value must be at least 2.");
        this.p = p;

        normMatrixOperation = new NormMatrixOperation(argument1.getRows(), argument1.getColumns(), p);
        normGradientMatrixOperation = new BinaryMatrixOperation(argument1.getRows(), argument1.getColumns(), (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> Math.pow(Math.abs(value1) / value2, p - 1) * Math.signum(value1));
        multiplyMatrixOperation = new BinaryMatrixOperation(argument1.getRows(), argument1.getColumns(), (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 * value2);
    }

    /**
     * Returns true is expression is executed as single step otherwise false.
     *
     * @return true is expression is executed as single step otherwise false.
     */
    protected boolean executeAsSingleStep() {
        return false;
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
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int sampleIndex) throws MatrixException {
        checkArgument(argument1, sampleIndex);
        result.setMatrix(sampleIndex, argument1.getMatrix(sampleIndex).constantAsMatrix(normMatrixOperation.apply(argument1.getMatrix(sampleIndex))));
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
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int sampleIndex) throws MatrixException {
        checkResultGradient(result, sampleIndex);
        // https://math.stackexchange.com/questions/1482494/derivative-of-the-l-p-norm/1482525
        if (!argument1.isStopGradient()) {
            Matrix normGradientMatrix = normGradientMatrixOperation.apply(argument1.getMatrix(sampleIndex), result.getMatrix(sampleIndex), argument1.getNewMatrix());
            Matrix resultMatrix = multiplyMatrixOperation.apply(result.getGradient(sampleIndex), normGradientMatrix, argument1.getNewMatrix());
            argument1.cumulateGradient(sampleIndex, resultMatrix, false);
        }
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        print();
        System.out.println(getExpressionName() + "(" + p + ", " + argument1.getName() + ") = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(true, " * (ABS(" + argument1.getName() + ")" + " / " + result.getName() + ")^" + (p - 1) + " * SGN("  + argument1.getName() + ")");
    }

}
