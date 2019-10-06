/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package utils;

import java.io.Serializable;

/**
 * Class that describes single computable expression including gradient expression.
 *
 */
public class Expression implements Serializable {

    private static final long serialVersionUID = -3692842009210981254L;

    /**
     * Types of functions supported.
     *
     */
    public enum Type {
        UNIFUN,
        BIFUN,
        ADD,
        SUB,
        DOT,
        MUL,
        DIV
    }

    /**
     * Unique ID of expression.
     *
     */
    private final int expressionID;

    /**
     * Node for first argument.
     *
     */
    private Node arg1;

    /**
     * Node for second argument.
     *
     */
    private Node arg2 = null;

    /**
     * Node for result.
     *
     */
    private Node result;

    /**
     * UniFunction used.
     *
     */
    private UniFunction uniFunction;

    /**
     * BiFunction used.
     *
     */
    private BiFunction biFunction;

    /**
     * Lambda operation to calculate uniFunction.
     *
     */
    private Matrix.MatrixUniOperation uniFunctionPrime;

    /**
     * Lambda operation to calculate derivative of uniFunction.
     *
     */
    private Matrix.MatrixUniOperation uniDerivativePrime;

    /**
     * Lambda operation to calculate biFunction.
     *
     */
    private Matrix.MatrixBiOperation biFunctionPrime;

    /**
     * Lambda operation to calculate derivative of biFunction.
     *
     */
    private Matrix.MatrixBiOperation biDerivativePrime;

    /**
     * Type of expression.
     *
     */
    private Type type;

    /**
     * Constructor for expression.
     *
     * @param expressionID unique ID for expression.
     * @param arg1 first argument.
     * @param result result.
     * @param uniFunction UniFunction.
     */
    Expression(int expressionID, Node arg1, Node result, UniFunction uniFunction) {
        this.expressionID = expressionID;
        type = Type.UNIFUN;
        this.arg1 = arg1;
        this.result = result;
        this.uniFunction = uniFunction;
        this.uniFunctionPrime = null;
        this.uniDerivativePrime = null;
    }

    /**
     * Constructor for expression.
     *
     * @param expressionID unique ID for expression.
     * @param arg1 first argument.
     * @param result result for expression.
     * @param uniFunctionPrime prime matrix UniFunction.
     * @param uniDerivativePrime prime matrix UniFunction derivative.
     */
    public Expression(int expressionID, Node arg1, Node result, Matrix.MatrixUniOperation uniFunctionPrime, Matrix.MatrixUniOperation uniDerivativePrime) {
        this.expressionID = expressionID;
        type = Type.UNIFUN;
        this.arg1 = arg1;
        this.result = result;
        this.uniFunction = null;
        this.uniFunctionPrime = uniFunctionPrime;
        this.uniDerivativePrime = uniDerivativePrime;
    }

    /**
     * Constructor for expression.
     *
     * @param expressionID unique ID for expression.
     * @param arg1 first argument.
     * @param arg2 second argument.
     * @param result result of expression.
     * @param biFunction BiFunction.
     */
    public Expression(int expressionID, Node arg1, Node arg2, Node result, BiFunction biFunction) {
        this.expressionID = expressionID;
        type = Type.BIFUN;
        this.arg1 = arg1;
        this.arg2 = arg2;
        this.result = result;
        this.biFunction = biFunction;
        this.biFunctionPrime = null;
        this.biDerivativePrime = null;
    }

    /**
     * Constructor for expression.
     *
     * @param expressionID unique ID for expression.
     * @param arg1 first argument.
     * @param arg2 second argument.
     * @param result result of expression.
     * @param biFunctionPrime prime matrix biFunction.
     * @param biDerivativePrime prime matrix biFunction derivative.
     */
    public Expression(int expressionID, Node arg1, Node arg2, Node result, Matrix.MatrixBiOperation biFunctionPrime, Matrix.MatrixBiOperation biDerivativePrime) {
        this.expressionID = expressionID;
        type = Type.BIFUN;
        this.arg1 = arg1;
        this.arg2 = arg2;
        this.result = result;
        this.biFunction = null;
        this.biFunctionPrime = biFunctionPrime;
        this.biDerivativePrime = biDerivativePrime;
    }

    /**
     * Constructor for expression with basic operation (add, subtract, dot, multiply, divide).
     *
     * @param expressionID unique ID for expression.
     * @param arg1 first argument.
     * @param arg2 second argument.
     * @param result result of expression.
     * @param type type of expression.
     * @throws MatrixException throws exception if type is uni or bi function.
     */
    public Expression(int expressionID, Node arg1, Node arg2, Node result, Type type) throws MatrixException {
        this.expressionID = expressionID;
        if (type == Type.UNIFUN || type == Type.BIFUN) throw new MatrixException("Exception cannot be of type function (FUN)");
        this.arg1 = arg1;
        this.arg2 = arg2;
        this.result = result;
        this.type = type;
    }

    /**
     * Return expression ID
     *
     * @return expression ID
     */
    public int getExpressionID() {
        return expressionID;
    }

    /**
     * Gets first argument of expression.
     *
     * @return first argument of expression.
     */
    public Node getArg1() {
        return arg1;
    }

    /**
     * Gets second argument of expression.
     *
     * @return second argument of expression.
     */
    public Node getArg2() {
        return arg2;
    }

    /**
     * Gets result of expression.
     *
     * @return result of expression.
     */
    public Node getResult() {
        return result;
    }

    /**
     * Gets type of expression.
     *
     * @return type of expression.
     */
    public Type getType() {
        return type;
    }

    /**
     * Gets UniFunction of expression.
     *
     * @return UniFunction of expression.
     */
    public UniFunction getUniFunction() {
        return uniFunction;
    }

    /**
     * Gets UniFunction prime.
     *
     * @return UniFunction prime.
     */
    public Matrix.MatrixUniOperation getUniFunctionPrime() {
        return uniFunctionPrime;
    }

    /**
     * Gets UniFunction derivative prime.
     *
     * @return UniFunction derivative prime.
     */
    public Matrix.MatrixUniOperation getUniDerivativePrime() {
        return uniDerivativePrime;
    }

    /**
     * Gets BiFunction of expression.
     *
     * @return BiFunction of expression.
     */
    public BiFunction getBiFunction() {
        return biFunction;
    }

    /**
     * Gets BiFunction prime.
     *
     * @return BiFunction prime.
     */
    public Matrix.MatrixBiOperation getBiFunctionPrime() {
        return biFunctionPrime;
    }

    /**
     * Gets BiFunction derivative prime.
     *
     * @return BiFunction derivative prime.
     */
    public Matrix.MatrixBiOperation getBiDerivativePrime() {
        return biDerivativePrime;
    }


    /**
     * Resets nodes of expression.
     *
     * @throws MatrixException throws exception if mandatory first argument or result are not defined.
     */
    public void resetExpression() throws MatrixException {
        if (arg1 != null) arg1.resetNode();
        else throw new MatrixException("First argument not defined.");
        if (arg2 != null) arg2.resetNode();
        if (result != null) result.resetNode();
        else throw new MatrixException("Result not defined.");
    }

    /**
     * Resets nodes of expression for specific data index.
     *
     * @param index data index.
     * @throws MatrixException throws exception if mandatory first argument or result are not defined.
     */
    public void resetExpression(int index) throws MatrixException {
        if (arg1 != null) arg1.resetNode(index);
        else throw new MatrixException("First argument not defined.");
        if (arg2 != null) arg2.resetNode(index);
        if (result != null) result.resetNode(index);
        else throw new MatrixException("Result not defined.");
    }

    /**
     * Calculates expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int index) throws MatrixException {
        switch (type) {
            case UNIFUN:
                if (arg1.getMatrix(index) == null) throw new MatrixException("Argument for uni operation not defined");
                if (uniFunction != null) result.setMatrix(index, uniFunction.applyFunction(arg1.getMatrix(index)));
                else result.setMatrix(index, arg1.getMatrix(index).apply(uniFunctionPrime));
                break;
            case BIFUN:
                if (arg1.getMatrix(index) == null || arg2.getMatrix(index) == null) throw new MatrixException("Arguments for bi operation not defined");
                if (uniFunction != null) result.setMatrix(index, biFunction.applyFunction(arg1.getMatrix(index), arg2.getMatrix(index)));
                else result.setMatrix(index, arg1.getMatrix(index).applyBi(arg2.getMatrix(index), biFunctionPrime));
                break;
            case ADD:
                if (arg1.getMatrix(index) == null || arg2.getMatrix(index) == null) throw new MatrixException("Arguments for ADD operation not defined");
                result.setMatrix(index, arg1.getMatrix(index).add(arg2.getMatrix(index)));
                break;
            case SUB:
                if (arg1.getMatrix(index) == null || arg2.getMatrix(index) == null) throw new MatrixException("Arguments for SUB operation not defined");
                result.setMatrix(index, arg1.getMatrix(index).subtract(arg2.getMatrix(index)));
                break;
            case DOT:
                if (arg1.getMatrix(index) == null || arg2.getMatrix(index) == null) throw new MatrixException("Arguments for DOT operation not defined");
                result.setMatrix(index, arg1.getMatrix(index).dot(arg2.getMatrix(index)));
                break;
            case MUL:
                if (arg1.getMatrix(index) == null || arg2.getMatrix(index) == null) throw new MatrixException("Arguments for MUL operation not defined");
                result.setMatrix(index, arg1.getMatrix(index).multiply(arg2.getMatrix(index)));
                break;
            case DIV:
                if (arg1.getMatrix(index) == null || arg2.getMatrix(index) == null) throw new MatrixException("Arguments for DIV operation not defined");
                result.setMatrix(index, arg1.getMatrix(index).divide(arg2.getMatrix(index)));
                break;
        }
    }

    /**
     * Calculates gradient of expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int index) throws MatrixException {
        if (result.getGradient(index) == null) throw new MatrixException("Result gradient not defined.");
        switch (type) {
            case UNIFUN:
                Matrix inUniGradient;
                if (uniFunction != null) inUniGradient = uniFunction.applyGradient(result.getMatrix(index), result.getGradient(index));
                else inUniGradient = result.getGradient(index).multiply(result.getMatrix(index).apply(uniDerivativePrime));
                arg1.updateGradient(index, inUniGradient, true);
                break;
            case BIFUN:
                Matrix inBiGradient;
                if (biFunction != null) inBiGradient = biFunction.applyGradient(result.getMatrix(index), arg2.getMatrix(index), result.getGradient(index));
                else inBiGradient = result.getGradient(index).multiply(result.getMatrix(index).applyBi(arg2.getMatrix(index), biDerivativePrime));
                arg1.updateGradient(index, inBiGradient, true);
                break;
            case ADD:
                arg1.updateGradient(index, result.getGradient(index), true);
                arg2.updateGradient(index, result.getGradient(index), true);
                break;
            case SUB:
                arg1.updateGradient(index, result.getGradient(index), true);
                arg2.updateGradient(index, result.getGradient(index), false);
                break;
            case DOT:
                arg1.updateGradient(index, result.getGradient(index).dot(arg2.getMatrix(index).T()), true);
                arg2.updateGradient(index, arg1.getMatrix(index).T().dot(result.getGradient(index)), true);
                break;
            case MUL:
                arg1.updateGradient(index, result.getGradient(index).multiply(arg2.getMatrix(index)), true);
                arg2.updateGradient(index, arg1.getMatrix(index).multiply(result.getGradient(index)), true);
                break;
            case DIV:
                arg1.updateGradient(index, result.getGradient(index).divide(arg2.getMatrix(index)), true);
                arg2.updateGradient(index, result.getGradient(index).multiply(arg1.getMatrix(index)).divide(arg2.getMatrix(index).power(2)), false);
                break;
        }
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        switch (type) {
            case UNIFUN:
                System.out.print(type + " " + uniFunction.getType() + " " + arg1 + " " + arg2 + " " + result);
                break;
            case BIFUN:
                System.out.print(type + " " + biFunction.getType() + " " + arg1 + " " + arg2 + " " + result);
                break;
            default:
                System.out.print(type + " " + arg1 + " " + arg2 + " " + result);
                break;
        }
    }

}
