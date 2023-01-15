/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.matrix.operation.AveragePoolGradientMatrixOperation;
import utils.matrix.operation.AveragePoolMatrixOperation;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Implements expression for average pooling operation.<br>
 *
 */
public class AveragePoolExpression extends AbstractUnaryExpression implements Serializable {

    /**
     * Reference to average pool matrix operation.
     *
     */
    private final AveragePoolMatrixOperation averagePoolMatrixOperation;

    /**
     * Reference to average pool gradient matrix operation.
     *
     */
    private final AveragePoolGradientMatrixOperation averagePoolGradientMatrixOperation;

    /**
     * Constructor for average pool expression.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @param stride stride of pooling operation.
     * @param filterRowSize filter row size for operation.
     * @param filterColumnSize filter column size for operation.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public AveragePoolExpression(int expressionID, Node argument1, Node result, int stride, int filterRowSize, int filterColumnSize) throws MatrixException {
        super("AVERAGE_POOL", "AVERAGE_POOL", expressionID, argument1, result);

        averagePoolMatrixOperation = new AveragePoolMatrixOperation(result.getRows(), result.getColumns(),filterRowSize, filterColumnSize, stride);
        averagePoolGradientMatrixOperation = new AveragePoolGradientMatrixOperation(result.getRows(), result.getColumns(), filterRowSize, filterColumnSize, stride);
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
        averagePoolMatrixOperation.apply(argument1.getMatrix(sampleIndex), result.getNewMatrix(sampleIndex));
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
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, averagePoolGradientMatrixOperation.apply(result.getGradient(sampleIndex), argument1.getNewMatrix()), false);
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
        print();
        System.out.println(getArgument1PrefixName() + "_GRADIENT(" + result.getName() + ")" + getArgument1SumPostfix());
    }

}
