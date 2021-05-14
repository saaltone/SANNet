/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.operation.BinaryMatrixOperation;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Class that describes expression for multiply operation.<br>
 *
 */
public class MultiplyExpression extends AbstractBinaryExpression implements Serializable {

    /**
     * Reference to multiply matrix operation.
     *
     */
    private final BinaryMatrixOperation multiplyMatrixOperation;

    /**
     * Constructor for multiply operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of expression.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public MultiplyExpression(int expressionID, Node argument1, Node argument2, Node result) throws MatrixException {
        super("MULTIPLY", "*", expressionID, argument1, argument2, result);

        // Checks if there is need to broadcast or un-broadcast due to scalar matrix.
        int rows = !argument1.isScalar() ? argument1.getRows() : argument2.getRows();
        int columns = !argument1.isScalar() ? argument1.getColumns() : argument2.getColumns();

        multiplyMatrixOperation = new BinaryMatrixOperation(rows, columns, (Matrix.MatrixBinaryOperation & Serializable) (value1, value2) -> value1 * value2);
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
     * @param index data index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int index) throws MatrixException {
        if (argument1.getMatrix(index) == null || argument2.getMatrix(index) == null) throw new MatrixException(getExpressionName() + "Arguments for operation not defined");
        multiplyMatrixOperation.apply(argument1.getMatrix(index), argument2.getMatrix(index), result.getNewMatrix(index));
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
     * @param index data index.
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int index) throws MatrixException {
        if (result.getGradient(index) == null) throw new MatrixException(getExpressionName() + ": Result gradient not defined.");
        argument1.cumulateGradient(index, multiplyMatrixOperation.apply(result.getGradient(index), argument2.getMatrix(index), argument1.getEmptyMatrix()), false);
        argument2.cumulateGradient(index, multiplyMatrixOperation.apply(argument1.getMatrix(index), result.getGradient(index), argument2.getEmptyMatrix()), false);

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
        printArgument2Gradient(true, false, " " + getOperationSignature() + " " + argument1.getName());
    }

}
