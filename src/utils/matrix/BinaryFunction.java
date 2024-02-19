/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.matrix;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;

import java.io.Serial;
import java.io.Serializable;

/**
 * Implement two (binary) argument function.<br>
 * Provides calculation for both function and it's derivative.<br>
 * Functions supported are listed in related type enum.<br>
 */
public class BinaryFunction implements Serializable {

    @Serial
    private static final long serialVersionUID = -3313251812046572490L;

    /**
     * Lambda function to calculate function.
     *
     */
    private final Matrix.MatrixBinaryOperation function;

    /**
     * Lambda function to calculate derivative of function.
     *
     */
    private final Matrix.MatrixBinaryOperation derivative;

    /**
     * Defines type of function such as Sigmoid, ReLU.
     *
     */
    private final BinaryFunctionType binaryFunctionType;

    /**
     * Delta value for Huber loss.
     *
     */
    private double huberDelta = 1;

    /**
     * Margin value for hinge loss.
     *
     */
    private double hingeMargin = 1;

    /**
     * Constructor for custom binary function.
     *
     * @param function function.
     * @param derivative derivative of function.
     */
    public BinaryFunction(Matrix.MatrixBinaryOperation function, Matrix.MatrixBinaryOperation derivative) {
        this.binaryFunctionType = BinaryFunctionType.CUSTOM;
        this.function = function;
        this.derivative = derivative;
    }

    /**
     * Constructor for custom binary function.
     *
     * @param function function.
     */
    public BinaryFunction(Matrix.MatrixBinaryOperation function) {
        this.binaryFunctionType = BinaryFunctionType.CUSTOM;
        this.function = function;
        this.derivative = null;
    }

    /**
     * Constructor for binary function.
     *
     * @param binaryFunctionType type of function to be used.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public BinaryFunction(BinaryFunctionType binaryFunctionType) throws MatrixException, DynamicParamException {
        this (binaryFunctionType, null);
    }

    /**
     * Constructor for binary function.<br>
     * Supported parameters are:<br>
     *     - hingeMargin: default value for hinge margin 1.<br>
     *     - huberDelta: default value for huber delta 1.<br>
     *     - numberOfQuantiles: number of quantiles for quantile loss. Default value 10.<br>
     *
     * @param binaryFunctionType type of function to be used.
     * @param params parameters used for function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public BinaryFunction(BinaryFunctionType binaryFunctionType, String params) throws DynamicParamException, MatrixException {
        this.binaryFunctionType = binaryFunctionType;
        switch (binaryFunctionType) {
            case POW -> {
                function = (Matrix.MatrixBinaryOperation & Serializable) Math::pow;
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> constant * Math.pow(value, constant - 1);
            }
            case MAX -> {
                function = (Matrix.MatrixBinaryOperation & Serializable) Math::max;
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 1;
            }
            case MIN -> {
                function = (Matrix.MatrixBinaryOperation & Serializable) Math::min;
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 1;
            }
            case MEAN_SQUARED_ERROR -> {
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 0.5 * Math.pow(value - constant, 2);
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> (value - constant);
            }
            case MEAN_SQUARED_LOGARITHMIC_ERROR -> {
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> Math.pow(Math.log(constant + 1) - Math.log(value + 1), 2);
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> -2 * (Math.log(constant + 1) - Math.log(value + 1)) / (constant + 1);
            }
            case MEAN_ABSOLUTE_ERROR -> {
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> Math.abs(value - constant);
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> Math.signum(value - constant);
            }
            case MEAN_ABSOLUTE_PERCENTAGE_ERROR -> {
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> Math.abs((value - constant) / constant);
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> (value - constant) / (Math.abs(constant) * Math.abs(value - constant));
            }
            case CROSS_ENTROPY -> {
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> -(constant * Math.log(value));
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> -(constant / value);
            }
            case BINARY_CROSS_ENTROPY -> {
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> -(constant * Math.log(value) + (1 - constant) * Math.log(1 - value));
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> -(constant / value - (1 - constant) / (1 - value));
            }
            case KULLBACK_LEIBLER -> {
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> (constant * Math.log(constant) - constant * Math.log((value)));
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> -(constant / value);
            }
            case NEGATIVE_LOG_LIKELIHOOD -> {
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> -Math.log((value));
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> -1 / value;
            }
            case POISSON -> {
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> value - constant * Math.log(value);
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 1 - constant / value;
            }
            case HINGE -> {
                if (params != null) {
                    DynamicParam dynamicParam = new DynamicParam(params, "(hingeMargin:DOUBLE)");
                    if (dynamicParam.hasParam("hingeMargin")) hingeMargin = dynamicParam.getValueAsDouble("hingeMargin");
                }
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> hingeMargin - constant * value <= 0 ? 0 : hingeMargin - constant * value;
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> hingeMargin - constant * value <= 0 ? 0 : -constant;
            }
            case SQUARED_HINGE -> {
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 1 - constant * value <= 0 ? 0 : Math.pow(1 - constant * value, 2);
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 1 - constant * value <= 0 ? 0 : -2 * constant * (1 - constant * value);
            }
            case HUBER -> {
                if (params != null) {
                    DynamicParam dynamicParam = new DynamicParam(params, "(huberDelta:DOUBLE)");
                    if (dynamicParam.hasParam("huberDelta")) huberDelta = dynamicParam.getValueAsDouble("huberDelta");
                }
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> Math.abs(value - constant) <= huberDelta ? 0.5 * Math.pow(value - constant, 2) : huberDelta * (Math.abs(value - constant) - 0.5 * huberDelta);
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> Math.abs(value - constant) <= huberDelta ? value - constant : huberDelta * Math.signum(value - constant);
            }
            case DIRECT_GRADIENT -> {
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 0;
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> constant;
            }
            case POLICY_GRADIENT -> { // -Math.log(policy_value at i, t) * Q_value (or A_value) at i, t
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 0;
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> -Math.log(value) * constant;
            }
            case DQN_REG_LOSS -> {
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 0.1 * value + Math.pow(value - constant, 2);
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 0.1 + 2 * (value - constant);
            }
            case POLICY_VALUE, COS_SIM -> {
                function = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 0;
                derivative = (Matrix.MatrixBinaryOperation & Serializable) (value, constant) -> 0;
            }
            case CUSTOM -> throw new MatrixException("Custom function cannot defined with this constructor.");
            default -> throw new MatrixException("Unknown binary function.");
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
     * Returns name of binary function.
     *
     * @return name of binary function.
     */
    public String getName() {
        return binaryFunctionType.toString();
    }

}
