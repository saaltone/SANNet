/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.matrix;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.procedure.ProcedureFactory;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Defines single (unary) argument function class.<br>
 * Provides calculation for both function and it's derivative.<br>
 * <br>
 * Functions supported are listed in related type enum.<br>
 */
public class UnaryFunction implements Serializable {

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
     * Matrix populated with ones.
     *
     */
    private Matrix ones;

    /**
     * Identity matrix.
     *
     */
    private Matrix identity;


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
     *
     * @param unaryFunctionType type of function to be used.
     * @param params parameters as DynamicParam type for function.
     * @throws DynamicParamException throws exception if parameters are not properly given.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private void setFunction(UnaryFunctionType unaryFunctionType, String params) throws DynamicParamException, MatrixException {
        switch(unaryFunctionType) {
            case ABS:
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::abs;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value / Math.abs(value);
                return;
            case COS:
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::cos;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> -Math.sin(value);
                return;
            case COSH:
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::cosh;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) Math::sinh;
                return;
            case EXP:
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::exp;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) Math::exp;
                return;
            case LOG:
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::log;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 / value;
                return;
            case LOG10:
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::log10;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 / (Math.log(10) * value);
                return;
            case SGN:
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::signum;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 0;
                return;
            case SIN:
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::sin;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) Math::cos;
                return;
            case SINH:
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::sinh;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) Math::cosh;
                return;
            case SQRT:
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::sqrt;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 / (2 * Math.sqrt(value));
                return;
            case CBRT:
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::cbrt;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 / (3 * Math.cbrt(value * value));
                return;
            case MULINV:
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 / value;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> -1 / Math.pow(value, 2);
                return;
            case TAN:
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::tan;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 + Math.pow(Math.tan(value), 2);
                return;
            case TANH:
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::tanh;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 - Math.pow(Math.tanh(value), 2);
                return;
            case LINEAR:
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1;
                return;
            case SIGMOID:
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 / (1 + Math.exp(-value));
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.exp(value) / Math.pow(1 + Math.exp(value), 2);
                return;
            case SWISH:
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value / (1 + Math.exp(-value));
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.exp(value) * (Math.exp(value) + value + 1) / Math.pow(1 + Math.exp(value), 2);
                return;
            case HARDSIGMOID:
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.min(1, Math.max(0, 0.125 * value + 0.5));
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> (value < -4 || value > 4) ? 0 : 0.125;
                return;
            case BIPOLARSIGMOID:
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 2 / (1 + Math.exp(-value)) - 1;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 2 * Math.exp(value) / Math.pow(Math.exp(value) + 1, 2);
                return;
            case TANHSIG:
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 2 / (Math.exp(-2 * value) + 1) - 1;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 4 * Math.exp(2 * value) / Math.pow(Math.exp(2 * value) + 1, 2);
                return;
            case TANHAPPR:
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> (Math.exp(2 * value) - 1) / (Math.exp(2 * value) + 1);
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 4 * Math.exp(2 * value) / Math.pow((Math.exp(2 * value) + 1), 2);
                return;
            case HARDTANH:
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.min(1, Math.max(-1, 0.5 * value));
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> (value < -2 || value > 2) ? 0 : 0.5;
                return;
            case SOFTPLUS:
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.log(1 + Math.exp(value));
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> -2 * value * Math.exp(-1 * Math.pow(value, 2) / 2);
                return;
            case SOFTSIGN:
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value / (Math.abs(value) + 1);
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1 / Math.pow((Math.abs(value) + 1), 2);
                return;
            case RELU:
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
                return;
            case ELU:
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
                return;
            case SELU:
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
                return;
            case GELU:
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * Math.pow(value, 3))));
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * Math.pow(value, 3)))) + (value * (0.134145 * Math.pow(value, 2) + 1) * Math.pow(1 / Math.cosh((0.044715 * Math.pow(value, 3) + value) * Math.sqrt(2 / Math.PI)), 2)) / Math.sqrt(2 * Math.PI);
                return;
            case SOFTMAX:
                function = (Matrix.MatrixUnaryOperation & Serializable) Math::exp;
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> 1;
                return;
            case GAUSSIAN:
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.exp(-1 * Math.pow(value, 2) / 2);
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> -2 * value * Math.exp(-1 * Math.pow(value, 2) / 2);
                return;
            case SINACT:
                function = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value < -0.5 * Math.PI ? -1 : value > 0.5 * Math.PI ? 1 : Math.sin(value);
                derivative = (Matrix.MatrixUnaryOperation & Serializable) (value) -> value < -0.5 * Math.PI ? 0 : value > 0.5 * Math.PI ? 0 : Math.cos(value);
                return;
            case CUSTOM:
                throw new MatrixException("Custom function cannot defined with this constructor.");
            default:
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
     * Applies function to matrix (value)
     *
     * @param value matrix
     * @return resulted matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyFunction(Matrix value) throws MatrixException {
        return applyFunction(value, false);
    }

    /**
     * Applies function to matrix (value)
     *
     * @param value matrix
     * @param inplace if true function is applied in place.
     * @return resulted matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyFunction(Matrix value, boolean inplace) throws MatrixException {
        Matrix result = inplace ? value : value.getNewMatrix();
        value.apply(result, this);
        if (unaryFunctionType == UnaryFunctionType.SOFTMAX) {
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
        if (unaryFunctionType != UnaryFunctionType.SOFTMAX) return gradient.multiply(value.apply(derivative));
        else {
            ones = (ones != null && ones.getRows() == value.getRows()) ? ones : new DMatrix(value.getRows(), 1, Initialization.ONE);
            identity = (identity != null && identity.getRows() == value.getRows()) ? identity : new DMatrix(value.getRows(), value.getRows(), Initialization.IDENTITY);
            // Calculate diagonal entries of 1 - arg and other entries -out i.e. identity - arg
            // Finally dot result by gradient resulting into derivative
            return identity.subtract(value.dot(ones.transpose())).dot(gradient);
        }
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
