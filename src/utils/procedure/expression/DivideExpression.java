/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.BinaryFunction;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.operation.BinaryMatrixOperation;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Implements expression for division operation.<br>
 *
 */
public class DivideExpression extends AbstractBinaryExpression {

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
        super("DIVIDE", expressionID, argument1, argument2, result);

        // Checks if there is need to broadcast or un-broadcast due to scalar matrix.
        int rows = !argument1.isScalar() ? argument1.getRows() : argument2.getRows();
        int columns = !argument1.isScalar() ? argument1.getColumns() : argument2.getColumns();

        divideMatrixOperation = new BinaryMatrixOperation(rows, columns, argument1.getDepth(), new BinaryFunction((Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 / value2));
        multiplyMatrixOperation = new BinaryMatrixOperation(rows, columns, argument1.getDepth(), new BinaryFunction((Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 * value2));
        divideGradientMatrixOperation = new BinaryMatrixOperation(rows, columns, argument1.getDepth(), new BinaryFunction((Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 / (value2 * value2)));
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
        return divideMatrixOperation.applyFunction(argument1Matrix, argument2Matrix);
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
        return divideMatrixOperation.applyFunction(resultGradient, argument2Matrix);
    }

    /**
     * Calculates argument 2 gradient matrix.
     *
     * @param sampleIndex     sample index.
     * @param resultGradient  result gradient.
     * @param argument1Matrix argument 1 matrix.
     * @param argument2Matrix argument 2 matrix.
     * @param resultMatrix    result matrix.
     * @return argument1 gradient matrix.
     * @throws MatrixException throws exception if calculation fails.
     */
    protected Matrix calculateArgument2Gradient(int sampleIndex, Matrix resultGradient, Matrix argument1Matrix, Matrix argument2Matrix, Matrix resultMatrix) throws MatrixException {
        return divideGradientMatrixOperation.applyFunction(multiplyMatrixOperation.applyFunction(resultGradient, argument1Matrix), argument2Matrix);
    }

    /**
     * Returns expression operation signature.
     *
     * @return expression operation signature.
     */
    protected String getExpressionOperationSignature() {
        return getArgument1().getName() + " / " + getArgument2().getName();
    }

    /**
     * Returns gradient 1 operation signature.
     *
     * @return gradient 1 operation signature.
     */
    protected String getGradientOperation1Signature() {
        return "d" + getResult().getName() + " / " + getArgument2().getName();
    }

    /**
     * Returns gradient 2 operation signature.
     *
     * @return gradient 2 operation signature.
     */
    protected String getGradientOperation2Signature() {
        return "d" + getResult().getName() + " * " + getArgument1().getName() + " / " + getArgument2().getName() + "^2";
    }

}
