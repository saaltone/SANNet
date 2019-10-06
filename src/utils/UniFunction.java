/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package utils;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Defines single argument function class.<br>
 * Provides calculation for both function and it's derivative.<br>
 * <br>
 * Functions supported are listed in related type enum.<br>
 */
public class UniFunction implements Serializable {

    private static final long serialVersionUID = -726049214135549929L;

    /**
     * Lambda function to calculate function.
     */
    private Matrix.MatrixUniOperation function;

    /**
     * Lambda function to calculate derivative of function.
     */
    private Matrix.MatrixUniOperation derivative;

    /**
     * Defines type of function such as Sigmoid, ReLU.
     */
    private UniFunctionType uniFunctionType;

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
     * Matrix populated with ones.
     *
     */
    private Matrix ones;

    /**
     * Identity matrix.
     *
     */
    private Matrix I;

    /**
     * Constructor for UniFunction.
     *
     * @param uniFunctionType type of function to be used.
     */
    public UniFunction(UniFunctionType uniFunctionType) {
        try {
            setFunction(uniFunctionType, null);
        } catch (DynamicParamException exception) {}
    }

    /**
     * Constructor for UniFunction.<br>
     * Supported parameters are:<br>
     *     - threshold: default value for RELU 0, for ELU 0, for SELU 0.<br>
     *     - alpha: default value for RELU 0, for ELU 1, for SELU 1.6732.<br>
     *     - lambda: default value for SELU 1.0507.<br>
     *
     * @param uniFunctionType type of function to be used.
     * @param params parameters used for function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public UniFunction(UniFunctionType uniFunctionType, String params) throws DynamicParamException {
        setFunction(uniFunctionType, params);
    }

    /**
     * Sets function with parameters.<br>
     * <br>
     * Supported parameters are:<br>
     *     - threshold: default value for RELU 0, for ELU 0, for SELU 0.<br>
     *     - alpha: default value for RELU 0, for ELU 1, for SELU 1.6732.<br>
     *     - lambda: default value for SELU 1.0507.<br>
     *
     * @param uniFunctionType type of function to be used.
     * @param params parameters as DynamicParam type for function.
     * @throws DynamicParamException throws exception if parameters are not properly given.
     */
    private void setFunction(UniFunctionType uniFunctionType, String params) throws DynamicParamException {
        this.uniFunctionType = uniFunctionType;
        switch(uniFunctionType) {
            case ABS:
                function = (Matrix.MatrixUniOperation & Serializable) Math::abs;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> value != 0 ? value / Math.abs(value) : Double.MAX_VALUE;
                return;
            case COS:
                function = (Matrix.MatrixUniOperation & Serializable) Math::cos;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> -Math.sin(value);
                return;
            case COSH:
                function = (Matrix.MatrixUniOperation & Serializable) Math::cosh;
                derivative = (Matrix.MatrixUniOperation & Serializable) Math::sinh;
                return;
            case EXP:
                function = (Matrix.MatrixUniOperation & Serializable) Math::exp;
                derivative = (Matrix.MatrixUniOperation & Serializable) Math::exp;
                return;
            case LOG:
                function = (Matrix.MatrixUniOperation & Serializable) Math::log;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> value != 0 ? 1 / value : Double.MAX_VALUE;
                return;
            case LOG10:
                function = (Matrix.MatrixUniOperation & Serializable) Math::log10;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> value != 0 ? 1 / (Math.log(10) * value) : Double.MAX_VALUE;
                return;
            case SGN:
                function = (Matrix.MatrixUniOperation & Serializable) Math::signum;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> value != 0 ? 0 : Double.MAX_VALUE;
                return;
            case SIN:
                function = (Matrix.MatrixUniOperation & Serializable) Math::sin;
                derivative = (Matrix.MatrixUniOperation & Serializable) Math::cos;
                return;
            case SINH:
                function = (Matrix.MatrixUniOperation & Serializable) Math::sinh;
                derivative = (Matrix.MatrixUniOperation & Serializable) Math::cosh;
                return;
            case SQRT:
                function = (Matrix.MatrixUniOperation & Serializable) Math::sqrt;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> value != 0 ? 1 / (2 * Math.sqrt(value)) : Double.MAX_VALUE;
                return;
            case CBRT:
                function = (Matrix.MatrixUniOperation & Serializable) Math::cbrt;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> value != 0 ? 1 / (3 * Math.cbrt(value * value)) : Double.MAX_VALUE;
                return;
            case MULINV:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> value != 0 ? 1 / value : Double.MAX_VALUE;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> value != 0 ? -1 / Math.pow(value, 2) : Double.MAX_VALUE;
                return;
            case TAN:
                function = (Matrix.MatrixUniOperation & Serializable) Math::tan;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 1 + Math.pow(Math.tan(value), 2);
                return;
            case TANH:
                function = (Matrix.MatrixUniOperation & Serializable) Math::tanh;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 1 - Math.pow(Math.tanh(value), 2);
                return;
            case LINEAR:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> value;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 1;
                return;
            case SIGMOID:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> 1 / (1 + Math.exp(-value));
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> Math.exp(value) / Math.pow(1 + Math.exp(value), 2);
                return;
            case SWISH:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> value / (1 + Math.exp(-value));
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> Math.exp(value) * (Math.exp(value) + value + 1) / Math.pow(1 + Math.exp(value), 2);
                return;
            case HARDSIGMOID:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> Math.min(1, Math.max(0, 0.125 * value + 0.5));
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> (value < -4 || value > 4) ? 0 : 0.125;
                return;
            case BIPOLARSIGMOID:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> 2 / (1 + Math.exp(-value)) - 1;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 2 * Math.exp(value) / Math.pow(Math.exp(value) + 1, 2);
                return;
            case TANSIG:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> 2 / (Math.exp(-2 * value) + 1) - 1;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 4 * Math.exp(2 * value) / Math.pow(Math.exp(2 * value) + 1, 2);
                return;
            case TANHAPPR:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> (Math.exp(2 * value) - 1) / (Math.exp(2 * value) + 1);
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 4 * Math.exp(2 * value) / Math.pow((Math.exp(2 * value) + 1), 2);
                return;
            case HARDTANH:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> Math.min(1, Math.max(-1, 0.5 * value));
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> (value < -2 || value > 2) ? 0 : 0.5;
                return;
            case SOFTPLUS:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> Math.log(1 + Math.exp(value));
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> -2 * value * Math.exp(-1 * Math.pow(value, 2) / 2);
                return;
            case SOFTSIGN:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> value / (Math.abs(value) + 1);
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 1 / Math.pow((Math.abs(value) + 1), 2);
                return;
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
                return;
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
                return;
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
                return;
            case GELU:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * Math.pow(value, 3))));
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * Math.pow(value, 3)))) + (value * (0.134145 * Math.pow(value, 2) + 1) * Math.pow(1 / Math.cosh((0.044715 * Math.pow(value, 3) + value) * Math.sqrt(2 / Math.PI)), 2)) / Math.sqrt(2 * Math.PI);
                return;
            case SOFTMAX:
                function = (Matrix.MatrixUniOperation & Serializable) Math::exp;
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> 1;
                return;
            case GAUSSIAN:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> Math.exp(-1 * Math.pow(value, 2) / 2);
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> -2 * value * Math.exp(-1 * Math.pow(value, 2) / 2);
                return;
            case SINACT:
                function = (Matrix.MatrixUniOperation & Serializable) (value) -> value < -0.5 * Math.PI ? -1 : value > 0.5 * Math.PI ? 1 : Math.sin(value);
                derivative = (Matrix.MatrixUniOperation & Serializable) (value) -> value < -0.5 * Math.PI ? 0 : value > 0.5 * Math.PI ? 0 : Math.cos(value);
                return;
            default:
        }
    }

    /**
     * Gets function.
     *
     * @return function used.
     */
    public Matrix.MatrixUniOperation getFunction() {
        return function;
    }

    /**
     * Gets derivative of function.
     *
     * @return derivative of function used.
     */
    public Matrix.MatrixUniOperation getDerivative() {
        return derivative;
    }

    /**
     * Gets function type used.
     *
     * @return function type used.
     */
    public UniFunctionType getType() {
        return uniFunctionType;
    }

    public Matrix applyFunction(Matrix value) throws MatrixException {
        return applyFunction(value, false);
    }

    public Matrix applyFunction(Matrix value, boolean inplace) throws MatrixException {
        Matrix result = inplace ? value : new DMatrix(value.getRows(), value.getCols());
        value.apply(result, this);
        if (uniFunctionType == UniFunctionType.SOFTMAX) {
            ProcedureFactory procedureFactory = result.getProcedureFactory();
            result.removeProcedureFactory();
            result = result.subtract(result.max()); // stable softmax X - max(X)
            result.apply(result, function);
            result.divide(result.sum(), result); // e^X / sum(e^X)
            result.setProcedureFactory(procedureFactory);
        }
        return result;
    }

    /**
     * Calculates inner gradient.
     *
     * @param value value for gradient calculation.
     * @param gradient outer gradient.
     * @return inner gradient
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyGradient(Matrix value, Matrix gradient) throws MatrixException {
        if (uniFunctionType != UniFunctionType.SOFTMAX) return gradient.multiply(value.apply(derivative));
        else {
            ones = (ones != null && ones.getRows() == value.getRows()) ? ones : new DMatrix(value.getRows(), 1, Init.ONE);
            I = (I != null && I.getRows() == value.getRows()) ? I : new DMatrix(value.getRows(), value.getRows(), Init.IDENTITY);
            // dFunc has diagonal entries of 1 - arg and other entries -out i.e. I - arg
            Matrix dFunc = I.subtract(value.dot(ones.T()));
            // Finally dFunc is dotted by gradient resulting into derivative
            return dFunc.dot(gradient);
        }
    }

}
