/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.operation.UnjoinMatrixOperation;
import utils.procedure.node.Node;

/**
 * Implements expression for unjoin function.<br>
 *
 */
public class UnjoinExpression extends AbstractUnaryExpression {

    /**
     * Unjoins at defined row.
     *
     */
    private final int unjoinAtRow;

    /**
     * Unjoins at defined column.
     *
     */
    private final int unjoinAtColumn;

    /**
     * Reference to unjoin matrix operation.
     *
     */
    private final UnjoinMatrixOperation unjoinMatrixOperation;

    /**
     * Constructor for unjoin function.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result.
     * @param unjoinAtRow unjoins at row.
     * @param unjoinAtColumn unjoins at column.
     * @param unjoinAtDepth unjoins at depth.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public UnjoinExpression(int expressionID, Node argument1, Node result, int unjoinAtRow, int unjoinAtColumn, int unjoinAtDepth) throws MatrixException {
        super("UNARY_FUNCTION", "", expressionID, argument1, result);

        this.unjoinAtRow = unjoinAtRow;
        this.unjoinAtColumn = unjoinAtColumn;

        unjoinMatrixOperation = new UnjoinMatrixOperation(result.getRows(), result.getColumns(), result.getDepth(), unjoinAtRow, unjoinAtColumn, unjoinAtDepth);
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
        return unjoinMatrixOperation.apply(argument1Matrix);
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
     */
    protected Matrix calculateArgument1Gradient(int sampleIndex, Matrix resultGradient, Matrix argument1Matrix, Matrix argument2Matrix, Matrix resultMatrix) {
        return unjoinMatrixOperation.applyGradient(argument1Matrix, resultGradient);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        print();
        System.out.println(getExpressionName() + ": " + "UNJOIN(" + argument1.getName() + "[" + unjoinAtRow + "," + unjoinAtColumn + "]" +  ") = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(false, "UNJOIN_GRADIENT(d" + result.getName() + "[" + unjoinAtRow + "," + unjoinAtColumn + "]" + ")");
    }

}
