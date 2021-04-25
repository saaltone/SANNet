/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.procedure.expression;

import utils.matrix.*;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Class that describes expression for unary function.<br>
 *
 */
public class UnaryFunctionExpression extends AbstractUnaryExpression implements Serializable {

    /**
     * Unary function type.
     *
     */
    private final UnaryFunctionType unaryFunctionType;

    /**
     * UnaryFunction used.
     *
     */
    private final UnaryFunction unaryFunction;

    /**
     * Constructor for unary function.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result.
     * @param unaryFunction UnaryFunction.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public UnaryFunctionExpression(int expressionID, Node argument1, Node result, UnaryFunction unaryFunction) throws MatrixException {
        super("UNARY_FUNCTION", String.valueOf(unaryFunction.getType()), expressionID, argument1, result);
        this.unaryFunctionType = unaryFunction.getType();
        this.unaryFunction = unaryFunction;
    }

    /**
     * Returns unary function type.
     *
     * @return unary function type.
     */
    public UnaryFunctionType getUnaryFunctionType() {
        return unaryFunctionType;
    }

    /**
     * Returns UnaryFunction of expression.
     *
     * @return UnaryFunction of expression.
     */
    public UnaryFunction getUnaryFunction() {
        return unaryFunction;
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
        if (argument1.getMatrix(index) == null) throw new MatrixException(getExpressionName() + "Argument for operation not defined");
        result.setMatrix(index, unaryFunction.applyFunction(argument1.getMatrix(index)));
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
        argument1.cumulateGradient(index, unaryFunction.applyGradient(result.getMatrix(index), result.getGradient(index)), false);
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        print();
        System.out.println(getExpressionName() + ": " + unaryFunctionType + "(" + argument1.getName() + ") = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(true, " * " + unaryFunctionType + "_GRADIENT(" + result.getName() + ")");
    }

}
