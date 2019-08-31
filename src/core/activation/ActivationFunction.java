/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.activation;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Matrix;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Defines activation function class for neural network.<br>
 * Provides calculation for both function and it's derivative.<br>
 *
 */
public class ActivationFunction implements Serializable {

    private static final long serialVersionUID = 4302816456291628148L;

    /**
     * Lambda function to calculate activation function.
     */
    private Matrix.MatrixUniOperation function;

    /**
     * Lambda function to calculate activation derivative of function.
     */
    private Matrix.MatrixUniOperation derivative;

    /**
     * Defines type of activation function such as Sigmoid, ReLU.
     */
    private ActivationFunctionType functionType;

    /**
     * Stores threshold value for ReLU function.
     */
    private double RELU_threshold = 0;

    /**
     * Stores alpha value for ReLU function.
     */
    private double RELU_alpha = 0;

    /**
     * Stores threshold value for ELU function.
     */
    private double ELU_threshold = 0;

    /**
     * Stores alpha value for ELU function.
     */
    private double ELU_alpha = 1;

    /**
     * Stores threshold value for SELU function.
     */
    private double SELU_threshold = 0;

    /**
     * Stores alpha value for SELU function.
     */
    private double SELU_alpha = 1.6732;

    /**
     * Stores lambda value for SELU function.
     */
    private double SELU_lambda = 1.0507;

    /**
     * Constructor for activation function.
     *
     * @param functionType type of activation function to be used.
     */
    public ActivationFunction(ActivationFunctionType functionType) {
        try {
            setActivationFunction(functionType, null);
        } catch (DynamicParamException exception) {}
    }

    /**
     * Constructor for activation function.<br>
     * Supported parameters are:<br>
     *     - threshold: default value for RELU 0, for ELU 0, for SELU 0.<br>
     *     - alpha: default value for RELU 0, for ELU 1, for SELU 1.6732.<br>
     *     - lambda: default value for SELU 1.0507.<br>
     *
     * @param functionType type of activation function to be used.
     * @param params parameters used for activation function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ActivationFunction(ActivationFunctionType functionType, String params) throws DynamicParamException {
        setActivationFunction(functionType, params);
    }

    /**
     * Sets activation function with parameters.<br>
     * <br>
     * Supported parameters are:<br>
     *     - threshold: default value for RELU 0, for ELU 0, for SELU 0.<br>
     *     - alpha: default value for RELU 0, for ELU 1, for SELU 1.6732.<br>
     *     - lambda: default value for SELU 1.0507.<br>
     *
     * @param functionType type of activation function to be used.
     * @param params parameters as DynamicParam type for activation function.
     * @throws DynamicParamException throws exception if parameters are not properly given.
     */
    private void setActivationFunction(ActivationFunctionType functionType, String params) throws DynamicParamException {
        this.functionType = functionType;
        switch(functionType) {
            case LINEAR:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> value;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 1;
                break;
            case SIGMOID:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> 1 / (1 + Math.exp(-value));
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> value * (1 - value);
                break;
            case HARDSIGMOID:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> Math.min(1, Math.max(0, 0.125 * value + 0.5));
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> (value < -4 || value > 4) ? 0 : 0.125;
                break;
            case BIPOLARSIGMOID:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> 2 / (1 + Math.exp(-value)) - 1;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 2 * Math.exp(value) / Math.pow(Math.exp(value) + 1, 2);
                break;
            case EXPONENTIAL:
                function = (Matrix.MatrixUniOperation & Serializable) Math::exp;
                derivative = (Matrix.MatrixUniOperation & Serializable) Math::exp;
                break;
            case TANH:
                function = (Matrix.MatrixUniOperation & Serializable) Math::tanh;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 1 - Math.pow(Math.tanh(value), 2);
                break;
            case TANSIG:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> 2 / (Math.exp(-2 * value) + 1) - 1;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 4 * Math.exp(2 * value) / Math.pow(Math.exp(2 * value) + 1, 2);
                break;
            case TANHAPPR:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> (Math.exp(2 * value) - 1) / (Math.exp(2 * value) + 1);
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 4 * Math.exp(2 * value) / Math.pow((Math.exp(2 * value) + 1), 2);
                break;
            case HARDTANH:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> Math.min(1, Math.max(-1, 0.5 * value));
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> (value < -2 || value > 2) ? 0 : 0.5;
                break;
            case SOFTPLUS:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> Math.log(1 + Math.exp(value));
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> -2 * value * Math.exp(-1 * Math.pow(value, 2) / 2);
                break;
            case SOFTSIGN:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> value / (Math.abs(value) + 1);
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 1 / Math.pow((Math.abs(value) + 1), 2);
                break;
            case RELU:
                if (params != null) {
                    HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
                    paramDefs.put("threshold", DynamicParam.ParamType.DOUBLE);
                    paramDefs.put("alpha", DynamicParam.ParamType.DOUBLE);
                    DynamicParam dParams = new DynamicParam(params, paramDefs);
                    if (dParams.hasParam("threshold")) RELU_threshold = dParams.getValueAsDouble("threshold");
                    if (dParams.hasParam("alpha")) RELU_alpha = dParams.getValueAsDouble("alpha");
                }
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> value < RELU_threshold ? RELU_alpha * value : value;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> value < RELU_threshold ? RELU_alpha : 1;
                break;
            case ELU:
                if (params != null) {
                    HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
                    paramDefs.put("threshold", DynamicParam.ParamType.DOUBLE);
                    paramDefs.put("alpha", DynamicParam.ParamType.DOUBLE);
                    DynamicParam dParams = new DynamicParam(params, paramDefs);
                    if (dParams.hasParam("threshold")) ELU_threshold = dParams.getValueAsDouble("threshold");
                    if (dParams.hasParam("alpha")) ELU_alpha = dParams.getValueAsDouble("alpha");
                }
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> value < ELU_threshold ? ELU_alpha * (Math.exp(value) - 1) : value;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> value < ELU_threshold ? ELU_alpha * Math.exp(value) : 1;
                break;
            case SELU:
                if (params != null) {
                    HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
                    paramDefs.put("threshold", DynamicParam.ParamType.DOUBLE);
                    paramDefs.put("alpha", DynamicParam.ParamType.DOUBLE);
                    paramDefs.put("lambda", DynamicParam.ParamType.DOUBLE);
                    DynamicParam dParams = new DynamicParam(params, paramDefs);
                    if (dParams.hasParam("threshold")) SELU_threshold = dParams.getValueAsDouble("threshold");
                    if (dParams.hasParam("alpha")) SELU_alpha = dParams.getValueAsDouble("alpha");
                    if (dParams.hasParam("lambda")) SELU_lambda = dParams.getValueAsDouble("lambda");
                }
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> value < SELU_threshold ? SELU_lambda * SELU_alpha * (Math.exp(value) - 1) : SELU_lambda * value;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> value < SELU_threshold ? SELU_lambda * SELU_alpha * Math.exp(value) : SELU_lambda;
                break;
            case GELU:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * Math.pow(value, 3))));
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * Math.pow(value, 3)))) + (value * (0.134145 * Math.pow(value, 2) + 1) * Math.pow(1 / Math.cosh((0.044715 * Math.pow(value, 3) + value) * Math.sqrt(2 / Math.PI)), 2)) / Math.sqrt(2 * Math.PI);
                break;
            case SOFTMAX:
                function = (Matrix.MatrixUniOperation & Serializable) Math::exp;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 1;
                break;
            case GAUSSIAN:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> Math.exp(-1 * Math.pow(value, 2) / 2);
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> -2 * value * Math.exp(-1 * Math.pow(value, 2) / 2);
                break;
            case SIN:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> value < -0.5 * Math.PI ? -1 : value > 0.5 * Math.PI ? 1 : Math.sin(value);
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> value < -0.5 * Math.PI ? 0 : value > 0.5 * Math.PI ? 0 : Math.cos(value);
//                function = (Matrix.MatrixUniOperation & Serializable) (value) -> Math.sin(value);
//                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> Math.cos(value);
                break;
            default:
                break;
        }
    }

    /**
     * Gets activation function.
     *
     * @return activation function used.
     */
    public Matrix.MatrixUniOperation getFunction() {
        return function;
    }

    /**
     * Gets derivative of activation function.
     *
     * @return derivative of activation function used.
     */
    public Matrix.MatrixUniOperation getDerivative() {
        return derivative;
    }

    /**
     * Gets activation function type used.
     *
     * @return activation function type used.
     */
    public ActivationFunctionType getType() {
        return functionType;
    }

}

