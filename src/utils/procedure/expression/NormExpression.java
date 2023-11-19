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
public class NormExpression extends AbstractUnaryExpression {

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

        normMatrixOperation = new NormMatrixOperation(argument1.getRows(), argument1.getColumns(), argument1.getDepth(), p);
        normGradientMatrixOperation = new BinaryMatrixOperation(argument1.getRows(), argument1.getColumns(), argument1.getDepth(), new BinaryFunction((Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> Math.pow(Math.abs(value1) / value2, p - 1) * Math.signum(value1)));
        multiplyMatrixOperation = new BinaryMatrixOperation(argument1.getRows(), argument1.getColumns(), argument1.getDepth(), new BinaryFunction((Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 * value2));
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
     * Resets expression.
     *
     */
    public void applyReset() {
    }

    /**
     * Calculates result matrix.
     *
     * @return result matrix.
     */
    protected Matrix calculateResult() {
        return null;
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
        return argument1Matrix.constantAsMatrix(normMatrixOperation.apply(argument1Matrix));
    }

    /**
     * Calculates argument 1 gradient matrix.
     */
    protected void calculateArgument1Gradient() {
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
        // https://math.stackexchange.com/questions/1482494/derivative-of-the-l-p-norm/1482525
        return multiplyMatrixOperation.applyFunction(resultGradient, normGradientMatrixOperation.applyFunction(argument1Matrix, resultMatrix));
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
