/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package utils.matrix;

import utils.DynamicParam;
import utils.DynamicParamException;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Defines two (binary) argument function class.<br>
 * Provides calculation for both function and it's derivative.<br>
 * <br>
 * Functions supported are listed in related type enum.
 */
public class BinaryFunction implements Serializable {

    private static final long serialVersionUID = -3313251812046572490L;

    /**
     * Lambda function to calculate function.
     */
    private Matrix.MatrixBinaryOperation function;

    /**
     * Lambda function to calculate derivative of function.
     */
    private Matrix.MatrixBinaryOperation derivative;

    /**
     * Defines type of function such as Sigmoid, ReLU.
     */
    private BinaryFunctionType binaryFunctionType;

    /**
     * Alpha value for Huber loss.
     *
     */
    private double huber_delta = 1;

    /**
     * Margin value for Hinge loss.
     *
     */
    private double hinge_margin = 1;

    /**
     * Constructor for BinaryFunction.
     *
     * @param binaryFunctionType type of function to be used.
     */
    public BinaryFunction(BinaryFunctionType binaryFunctionType) {
        try {
            setFunction(binaryFunctionType, null);
        } catch (DynamicParamException exception) {}
    }

    /**
     * Constructor for BinaryFunction.<br>
     * Supported parameters are:<br>
     *     - alpha: default value for Huber loss 1.<br>
     *     - hinge: default value for hinge margin 1.<br>
     *
     * @param binaryFunctionType type of function to be used.
     * @param params parameters used for function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public BinaryFunction(BinaryFunctionType binaryFunctionType, String params) throws DynamicParamException {
        setFunction(binaryFunctionType, params);
    }

    /**
     * Sets function with parameters.<br>
     * <br>
     * Supported parameters are:<br>
     *     - alpha: default value for Huber loss 1.<br>
     *     - hinge: default value for hinge margin 1.<br>
     *
     * @param binaryFunctionType type of function to be used.
     * @param params parameters as DynamicParam type for function.
     * @throws DynamicParamException throws exception if parameters are not properly given.
     */
    private void setFunction(BinaryFunctionType binaryFunctionType, String params) throws DynamicParamException {
        this.binaryFunctionType = binaryFunctionType;
        switch(binaryFunctionType) {
            case POW:
                function = (Matrix.MatrixBinaryOperation & Serializable) Math::pow;
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> constant * Math.pow(value, constant - 1);
                break;
            case MEAN_SQUARED_ERROR:
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 0.5 * Math.pow(value - constant, 2);
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> (value - constant);
                break;
            case MEAN_SQUARED_LOGARITHMIC_ERROR:
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> Math.pow(Math.log(constant + 1 > 0 ? constant + 1 : 10E-8) - Math.log(value + 1 > 0 ? value + 1 : 10E-8), 2);
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> -2 * (Math.log(constant + 1 > 0 ? constant + 1 : 10E-8) - Math.log(value + 1 > 0 ? value + 1 : 10E-8)) / (constant + 1 != 0 ? constant + 1 : Double.MAX_VALUE);
                break;
            case MEAN_ABSOLUTE_ERROR:
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> Math.abs(value - constant);
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> Math.signum(value - constant);
                break;
            case MEAN_ABSOLUTE_PERCENTAGE_ERROR:
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> constant != 0 ? 100 * Math.abs((value - constant) / constant) : Double.MAX_VALUE;
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 100 * (value - constant) / ((constant != 0 && (value - constant) != 0) ? Math.abs(constant) * Math.abs(value - constant) : Double.MAX_VALUE);
                break;
            case CROSS_ENTROPY:
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> -(constant * Math.log(value > 0 ? value : 10E-8));
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> -(constant / (value > 0 ? value : 10E-8));
                break;
            case KULLBACK_LEIBLER:
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> (constant * Math.log(constant > 0 ? constant : 10E-8) - constant * Math.log((value > 0 ? value : 10E-8)));
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> -(constant / (value > 0 ? value : 10E-8));
                break;
            case NEGATIVE_LOG_LIKELIHOOD:
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> -Math.log((value > 0 ? value : 10E-8));
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> -1 / (value > 0 ? value : 10E-8);
                break;
            case POISSON:
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> value - constant * Math.log((value > 0 ? value : 10E-8));
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 1 - constant / (value > 0 ? value : 10E-8);
                break;
            case HINGE:
                if (params != null) {
                    HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
                    paramDefs.put("margin", DynamicParam.ParamType.DOUBLE);
                    DynamicParam dParams = new DynamicParam(params, paramDefs);
                    if (dParams.hasParam("margin")) hinge_margin = dParams.getValueAsDouble("margin");
                }
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> hinge_margin - constant * value <= 0 ? 0 : hinge_margin - constant * value;
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> hinge_margin - constant * value <= 0 ? 0 : - constant;
                break;
            case SQUARED_HINGE:
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 1 - constant * value <= 0 ? 0 : Math.pow(1 - constant * value, 2);
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 1 - constant * value <= 0 ? 0 : - 2 * constant * (1 - constant * value);
                break;
            case HUBER:
                if (params != null) {
                    HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
                    paramDefs.put("delta", DynamicParam.ParamType.DOUBLE);
                    DynamicParam dParams = new DynamicParam(params, paramDefs);
                    if (dParams.hasParam("delta")) huber_delta = dParams.getValueAsDouble("delta");
                }
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> Math.abs(value - constant) <= huber_delta ? 0.5 * Math.pow(value - constant, 2) : huber_delta * Math.abs(value - constant) - 0.5 * Math.pow(huber_delta, 2);
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> Math.abs(value - constant) <= huber_delta ? value - constant : huber_delta * Math.signum(value - constant);
                break;
            default:
                break;
        }
    }

    /**
     * Returns function.
     *
     * @return function used.
     */
    public Matrix.MatrixBinaryOperation getFunction() {
        return function;
    }

    /**
     * Returns derivative of function.
     *
     * @return derivative of function used.
     */
    public Matrix.MatrixBinaryOperation getDerivative() {
        return derivative;
    }

    /**
     * Returns function type used.
     *
     * @return function type used.
     */
    public BinaryFunctionType getType() {
        return binaryFunctionType;
    }

    /**
     * Applies function to value and constant.
     *
     * @param value value
     * @param constant constant
     * @return applied function.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyFunction(Matrix value, Matrix constant) throws MatrixException {
        return applyFunction(value, constant, false);
    }

    /**
     * Applies function to value and constant.
     *
     * @param value value
     * @param constant constant
     * @param inplace if true function is applied inplace.
     * @return result of applied function.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyFunction(Matrix value, Matrix constant, boolean inplace) throws MatrixException {
        Matrix result = inplace ? value : new DMatrix(value.getRows(), value.getCols());
        value.applyBi(constant, result, this);
        return result;
    }

    /**
     * Calculates inner gradient.
     *
     * @param value value for inner gradient calculation.
     * @param constant constant value for inner gradient calculation.
     * @param gradient outer gradient value for inner gradient calculation.
     * @return inner gradient
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyGradient(Matrix value, Matrix constant, Matrix gradient) throws MatrixException {
        return gradient.multiply(value.applyBi(constant, this));
    }

}
