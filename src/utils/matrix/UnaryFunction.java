/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.matrix;

import utils.DynamicParam;
import utils.DynamicParamException;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashMap;

/**
 * Defines single (unary) argument function class.<br>
 * Provides calculation for both function and it's derivative.<br>
 * <br>
 * Functions supported are listed in related type enum.<br>
 */
public class UnaryFunction implements Serializable {

    @Serial
    private static final long serialVersionUID = -726049214135549929L;

    /**
     * Lambda function to calculate function.
     *
     */
    private Matrix.MatrixUnaryOperation function;

    /**
     * Lambda function to calculate derivative of function.
     *
     */
    private Matrix.MatrixUnaryOperation derivative;

    /**
     * Defines type of function such as Sigmoid, ReLU.
     *
     */
    private final UnaryFunctionType unaryFunctionType;

    /**
     * Stores threshold value for ReLU function.
     *
     */
    private double RELUThreshold = 0;

    /**
     * Stores alpha value for ReLU function.
     *
     */
    private double RELUAlpha = 0;

    /**
     * Stores threshold value for ELU function.
     *
     */
    private double ELUThreshold = 0;

    /**
     * Stores alpha value for ELU function.
     *
     */
    private double ELUAlpha = 1;

    /**
     * Stores threshold value for SELU function.
     *
     */
    private double SELUThreshold = 0;

    /**
     * Stores alpha value for SELU function.
     *
     */
    private double SELUAlpha = 1.6732;

    /**
     * Stores lambda value for SELU function.
     *
     */
    private double SELULambda = 1.0507;

    /**
     * Stores tau value for Gumbel Softmax.
     *
     */
    private double gumbelSoftmaxTau = 2.75;

    /**
     * Constructor for UnaryFunction to create custom function.
     *
     * @param function function.
     * @param derivative derivative of function.
     */
    public UnaryFunction(Matrix.MatrixUnaryOperation function, Matrix.MatrixUnaryOperation derivative) {
        this.unaryFunctionType = UnaryFunctionType.CUSTOM;
        this.function = function;
        this.derivative = derivative;
    }

    /**
     * Constructor for UnaryFunction.
     *
     * @param unaryFunctionType type of function to be used.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public UnaryFunction(UnaryFunctionType unaryFunctionType) throws DynamicParamException, MatrixException {
        this.unaryFunctionType = unaryFunctionType;
        setFunction(unaryFunctionType, null);
    }

    /**
     * Constructor for UnaryFunction.<br>
     * Supported parameters are:<br>
     *     - threshold: default value for RELU 0, for ELU 0, for SELU 0.<br>
     *     - alpha: default value for RELU 0, for ELU 1, for SELU 1.6732.<br>
     *     - lambda: default value for SELU 1.0507.<br>
     *     - tau: default value for Gumbel Softmax 2.65.<br>
     *
     * @param unaryFunctionType type of function to be used.
     * @param params parameters used for function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public UnaryFunction(UnaryFunctionType unaryFunctionType, String params) throws DynamicParamException, MatrixException {
        this.unaryFunctionType = unaryFunctionType;
        setFunction(unaryFunctionType, params);
    }

    /**
     * Sets function with parameters.<br>
     * <br>
     * Supported parameters are:<br>
     *     - threshold: default value for RELU 0, for ELU 0, for SELU 0.<br>
     *     - alpha: default value for RELU 0, for ELU 1, for SELU 1.6732.<br>
     *     - lambda: default value for SELU 1.0507.<br>
     *     - tau: default value for Gumbel Softmax 5.<br>
     *     - tauDecay: default value for Gumbel Softmax 0.999. <br>
     *
     * @param unaryFunctionType type of function to be used.
     * @param params parameters as DynamicParam type for function.
     * @throws DynamicParamException throws exception if parameters are not properly given.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private void setFunction(UnaryFunctionType unaryFunctionType, String params) throws DynamicParamException, MatrixException {
        switch (unaryFunctionType) {
            case ABS -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::abs;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value / Math.abs(value);
            }
            case COS -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::cos;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> -Math.sin(value);
            }
            case COSH -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::cosh;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) Math::sinh;
            }
            case EXP -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::exp;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) Math::exp;
            }
            case LOG -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::log;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 / value;
            }
            case LOG10 -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::log10;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 / (Math.log(10) * value);
            }
            case SGN -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::signum;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 0;
            }
            case SIN -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::sin;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) Math::cos;
            }
            case SINH -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::sinh;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) Math::cosh;
            }
            case SQRT -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::sqrt;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 / (2 * Math.sqrt(value));
            }
            case CBRT -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::cbrt;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 / (3 * Math.cbrt(value * value));
            }
            case MULINV -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 / value;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> -1 / Math.pow(value, 2);
            }
            case TAN -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::tan;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 + Math.pow(Math.tan(value), 2);
            }
            case TANH -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::tanh;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 - Math.pow(Math.tanh(value), 2);
            }
            case LINEAR -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1;
            }
            case SIGMOID -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 / (1 + Math.exp(-value));
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.exp(value) / Math.pow(1 + Math.exp(value), 2);
            }
            case SWISH -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value / (1 + Math.exp(-value));
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.exp(value) * (Math.exp(value) + value + 1) / Math.pow(1 + Math.exp(value), 2);
            }
            case HARDSIGMOID -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.min(1, Math.max(0, 0.125 * value + 0.5));
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> (value < -4 || value > 4) ? 0 : 0.125;
            }
            case BIPOLARSIGMOID -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 2 / (1 + Math.exp(-value)) - 1;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 2 * Math.exp(value) / Math.pow(Math.exp(value) + 1, 2);
            }
            case TANHSIG -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 2 / (Math.exp(-2 * value) + 1) - 1;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 4 * Math.exp(2 * value) / Math.pow(Math.exp(2 * value) + 1, 2);
            }
            case TANHAPPR -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> (Math.exp(2 * value) - 1) / (Math.exp(2 * value) + 1);
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 4 * Math.exp(2 * value) / Math.pow((Math.exp(2 * value) + 1), 2);
            }
            case HARDTANH -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.min(1, Math.max(-1, 0.5 * value));
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> (value < -2 || value > 2) ? 0 : 0.5;
            }
            case SOFTPLUS -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.log(1 + Math.exp(value));
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> -2 * value * Math.exp(-1 * Math.pow(value, 2) / 2);
            }
            case SOFTSIGN -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value / (Math.abs(value) + 1);
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 / Math.pow((Math.abs(value) + 1), 2);
            }
            case RELU -> {
                if (params != null) {
                    HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
                    paramDefs.put("threshold", DynamicParam.ParamType.DOUBLE);
                    paramDefs.put("alpha", DynamicParam.ParamType.DOUBLE);
                    DynamicParam dParams = new DynamicParam(params, paramDefs);
                    if (dParams.hasParam("threshold")) RELUThreshold = dParams.getValueAsDouble("threshold");
                    if (dParams.hasParam("alpha")) RELUAlpha = dParams.getValueAsDouble("alpha");
                }
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value < RELUThreshold ? RELUAlpha * value : value;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value < RELUThreshold ? RELUAlpha : 1;
            }
            case RELU_COS -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.max(0, value) + Math.cos(value);
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> (value < 0 ? 0 : 1) - Math.sin(value);
            }
            case RELU_SIN -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.max(0, value) + Math.sin(value);
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> (value < 0 ? 0 : 1) + Math.cos(value);
            }
            case ELU -> {
                if (params != null) {
                    HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
                    paramDefs.put("threshold", DynamicParam.ParamType.DOUBLE);
                    paramDefs.put("alpha", DynamicParam.ParamType.DOUBLE);
                    DynamicParam dParams = new DynamicParam(params, paramDefs);
                    if (dParams.hasParam("threshold")) ELUThreshold = dParams.getValueAsDouble("threshold");
                    if (dParams.hasParam("alpha")) ELUAlpha = dParams.getValueAsDouble("alpha");
                }
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value < ELUThreshold ? ELUAlpha * (Math.exp(value) - 1) : value;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value < ELUThreshold ? ELUAlpha * Math.exp(value) : 1;
            }
            case SELU -> {
                if (params != null) {
                    HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
                    paramDefs.put("threshold", DynamicParam.ParamType.DOUBLE);
                    paramDefs.put("alpha", DynamicParam.ParamType.DOUBLE);
                    paramDefs.put("lambda", DynamicParam.ParamType.DOUBLE);
                    DynamicParam dParams = new DynamicParam(params, paramDefs);
                    if (dParams.hasParam("threshold")) SELUThreshold = dParams.getValueAsDouble("threshold");
                    if (dParams.hasParam("alpha")) SELUAlpha = dParams.getValueAsDouble("alpha");
                    if (dParams.hasParam("lambda")) SELULambda = dParams.getValueAsDouble("lambda");
                }
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value < SELUThreshold ? SELULambda * SELUAlpha * (Math.exp(value) - 1) : SELULambda * value;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value < SELUThreshold ? SELULambda * SELUAlpha * Math.exp(value) : SELULambda;
            }
            case GELU -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * Math.pow(value, 3))));
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * Math.pow(value, 3)))) + (value * (0.134145 * Math.pow(value, 2) + 1) * Math.pow(1 / Math.cosh((0.044715 * Math.pow(value, 3) + value) * Math.sqrt(2 / Math.PI)), 2)) / Math.sqrt(2 * Math.PI);
            }
            case SOFTMAX -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1;
            }
            case GUMBEL_SOFTMAX -> {
                if (params != null) {
                    HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
                    paramDefs.put("tau", DynamicParam.ParamType.DOUBLE);
                    DynamicParam dParams = new DynamicParam(params, paramDefs);
                    if (dParams.hasParam("tau")) gumbelSoftmaxTau = dParams.getValueAsDouble("tau");
                }
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1;
            }
            case GAUSSIAN -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.exp(-1 * Math.pow(value, 2) / 2);
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> -2 * value * Math.exp(-1 * Math.pow(value, 2) / 2);
            }
            case SINACT -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value < -0.5 * Math.PI ? -1 : value > 0.5 * Math.PI ? 1 : Math.sin(value);
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value < -0.5 * Math.PI ? 0 : value > 0.5 * Math.PI ? 0 : Math.cos(value);
            }
            case LOGIT -> {
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.log(value / (1 - value));
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> -1 / ((value - 1) * value);
            }
            case CUSTOM -> throw new MatrixException("Custom function cannot be defined with this constructor.");
            default -> throw new MatrixException("Unknown unary function.");
        }
    }

    /**
     * Returns function.
     *
     * @return function used.
     */
    public Matrix.MatrixUnaryOperation getFunction() {
        return function;
    }

    /**
     * Returns derivative of function.
     *
     * @return derivative of function used.
     */
    public Matrix.MatrixUnaryOperation getDerivative() {
        return derivative;
    }

    /**
     * Returns function type used.
     *
     * @return function type used.
     */
    public UnaryFunctionType getType() {
        return unaryFunctionType;
    }

    /**
     * Return Gumbel softmax tau.
     *
     * @return Gumbel softmax tau.
     */
    public double getGumbelSoftmaxTau() {
        return gumbelSoftmaxTau;
    }

    /**
     * Applies function to matrix (value)
     *
     * @param first first matrix
     * @return resulted matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyFunction(Matrix first) throws MatrixException {
        Matrix result = first.getNewMatrix();
        switch (unaryFunctionType) {
            case SOFTMAX -> first.softmax(result);
            case GUMBEL_SOFTMAX -> first.gumbelSoftmax(result, gumbelSoftmaxTau);
            default -> first.apply(result, this);
        }
        return result;
    }

    /**
     * Calculates inner gradient.
     *
     * @param first first matrix.
     * @param outputGradient output gradient.
     * @return input gradient
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyGradient(Matrix first, Matrix outputGradient) throws MatrixException {
        return switch (unaryFunctionType) {
            case SOFTMAX, GUMBEL_SOFTMAX -> first.softmaxGrad().dot(outputGradient);
            default -> outputGradient.multiply(first.apply(derivative));
        };
    }

    /**
     * Returns name of unary function.
     *
     * @return name of unary function.
     */
    public String getName() {
        return unaryFunctionType.toString();
    }

}
