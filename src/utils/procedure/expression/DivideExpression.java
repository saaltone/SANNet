/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.operation.BinaryMatrixOperation;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Class that describes expression for division operation.<br>
 *
 */
public class DivideExpression extends AbstractBinaryExpression implements Serializable {

    /**
     * Reference to divide matrix operation.
     *
     */
    private final BinaryMatrixOperation divideMatrixOperation;

    /**
     * Reference to multiply matrix operation.
     *
     */
    private final BinaryMatrixOperation multiplyMatrixOperation;

    /**
     * Reference to divide gradient matrix operation.
     *
     */
    private final BinaryMatrixOperation divideGradientMatrixOperation;

    /**
     * Constructor for division operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public DivideExpression(int expressionID, Node argument1, Node argument2, Node result) throws MatrixException {
        super("DIVIDE", "/", expressionID, argument1, argument2, result);

        // Checks if there is need to broadcast or un-broadcast due to scalar matrix.
        int rows = !argument1.isScalar() ? argument1.getRows() : argument2.getRows();
        int columns = !argument1.isScalar() ? argument1.getColumns() : argument2.getColumns();

        divideMatrixOperation = new BinaryMatrixOperation(rows, columns, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 / value2);
        multiplyMatrixOperation = new BinaryMatrixOperation(rows, columns, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 * value2);
        divideGradientMatrixOperation = new BinaryMatrixOperation(rows, columns, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 / (value2 * value2));
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
        if (argument1.getMatrix(sampleIndex) == null || argument2.getMatrix(sampleIndex) == null) throw new MatrixException(getExpressionName() + ": Arguments for operation not defined");
        divideMatrixOperation.apply(argument1.getMatrix(sampleIndex), argument2.getMatrix(sampleIndex), result.getNewMatrix(sampleIndex));
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
        if (result.getGradient(sampleIndex) == null) throw new MatrixException(getExpressionName() + ": Result gradient not defined.");
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, divideMatrixOperation.apply(result.getGradient(sampleIndex), argument2.getMatrix(sampleIndex), argument1.getEmptyMatrix()), false);
        if (!argument2.isStopGradient()) {
            Matrix multiplyGradientResult = multiplyMatrixOperation.apply(result.getGradient(sampleIndex), argument1.getMatrix(sampleIndex), argument2.getEmptyMatrix());
            Matrix divideGradientResult = divideGradientMatrixOperation.apply(multiplyGradientResult, argument2.getMatrix(sampleIndex), argument1.getEmptyMatrix());
            argument2.cumulateGradient(sampleIndex, divideGradientResult, true);
        }
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
        printArgument1Gradient(true, " " + getOperationSignature() + " " + argument2.getName());
        printArgument2Gradient(true, false, " * " + argument1.getName() + " " + getOperationSignature() + " " + argument2.getName() + "^2");
    }

}
