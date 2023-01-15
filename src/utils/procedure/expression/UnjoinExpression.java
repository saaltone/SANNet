/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.matrix.operation.UnjoinMatrixOperation;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Implements expression for unjoin function.<br>
 *
 */
public class UnjoinExpression extends AbstractUnaryExpression implements Serializable {

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
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public UnjoinExpression(int expressionID, Node argument1, Node result, int unjoinAtRow, int unjoinAtColumn) throws MatrixException {
        super("UNARY_FUNCTION", "", expressionID, argument1, result);

        this.unjoinAtRow = unjoinAtRow;
        this.unjoinAtColumn = unjoinAtColumn;

        unjoinMatrixOperation = new UnjoinMatrixOperation(result.getRows(), result.getColumns(), unjoinAtRow, unjoinAtColumn);
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
        unjoinMatrixOperation.apply(argument1.getMatrix(sampleIndex), result.getNewMatrix(sampleIndex));
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
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, unjoinMatrixOperation.applyGradient(argument1.getMatrix(sampleIndex), result.getGradient(sampleIndex)), false);
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
