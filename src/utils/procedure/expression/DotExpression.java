/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.matrix.operation.DotMatrixOperation;
import utils.procedure.node.Node;

/**
 * Implements expression for dot operation.<br>
 *
 */
public class DotExpression extends AbstractBinaryExpression {

    /**
     * Reference to dot matrix operation.
     *
     */
    private final DotMatrixOperation dotMatrixOperation;

    /**
     * Reference to dot gradient 1 matrix operation.
     *
     */
    private final DotMatrixOperation dotGradient1MatrixOperation;

    /**
     * Reference to dot gradient 2 matrix operation.
     *
     */
    private final DotMatrixOperation dotGradient2MatrixOperation;

    /**
     * Constructor for dot operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public DotExpression(int expressionID, Node argument1, Node argument2, Node result) throws MatrixException {
        super("DOT", "x", expressionID, argument1, argument2, result);

        dotMatrixOperation = new DotMatrixOperation(argument1.getRows(), argument2.getRows(), argument2.getColumns(), argument1.getDepth());
        dotGradient1MatrixOperation = new DotMatrixOperation(result.getRows(), argument2.getColumns(), argument2.getRows(), argument2.getDepth());
        dotGradient2MatrixOperation = new DotMatrixOperation(argument1.getColumns(), result.getRows(), result.getColumns(), argument1.getDepth());
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
        checkArguments(argument1, argument2, sampleIndex);
        result.setMatrix(sampleIndex, dotMatrixOperation.apply(argument1.getMatrix(sampleIndex), argument2.getMatrix(sampleIndex)));
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
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, dotGradient1MatrixOperation.apply(result.getGradient(sampleIndex), argument2.getMatrix(sampleIndex).transpose()), false);
        if (!argument2.isStopGradient()) argument2.cumulateGradient(sampleIndex, dotGradient2MatrixOperation.apply(argument1.getMatrix(sampleIndex).transpose(), result.getGradient(sampleIndex)), false);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        printBasicBinaryExpression();
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(true, " " + getOperationSignature() + " " + argument2.getName() + ".T");
        printArgument2Gradient(false, false, argument1.getName() + ".T" + " " + getOperationSignature() + " " + getResultGradientName());
    }

}
