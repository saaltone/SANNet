/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.activation;

import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunction;
import utils.matrix.UnaryFunctionType;

/**
 * Implements activation function for neural network.
 *
 */
public class ActivationFunction extends UnaryFunction {

    /**
     * Constructor for activation function.
     *
     * @param activationFunctionType activation function type.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ActivationFunction(ActivationFunctionType activationFunctionType) throws MatrixException, DynamicParamException {
        this(activationFunctionType, null);
    }

    /**
     * Constructor for activation function.<br>
     * Supported parameters are:<br>
     *     - threshold: default value for RELU 0, for ELU 0, for SELU 0.<br>
     *     - alpha: default value for RELU 0, for ELU 1, for SELU 1.6732.<br>
     *     - lambda: default value for SELU 1.0507.<br>
     *     - tau: default value for (Gumbel) Softmax 1.<br>
     *
     * @param activationFunctionType activation function type.
     * @param params parameters used for activation function.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ActivationFunction(ActivationFunctionType activationFunctionType, String params) throws MatrixException, DynamicParamException {
        super(switch(activationFunctionType) {
            case LINEAR -> UnaryFunctionType.LINEAR;
            case SIGMOID -> UnaryFunctionType.SIGMOID;
            case SWISH -> UnaryFunctionType.SWISH;
            case HARDSIGMOID -> UnaryFunctionType.HARDSIGMOID;
            case BIPOLARSIGMOID -> UnaryFunctionType.BIPOLARSIGMOID;
            case STANH -> UnaryFunctionType.STANH;
            case TANH -> UnaryFunctionType.TANH;
            case TANHSIG -> UnaryFunctionType.TANHSIG;
            case TANHAPPR -> UnaryFunctionType.TANHAPPR;
            case HARDTANH -> UnaryFunctionType.HARDTANH;
            case EXP -> UnaryFunctionType.EXP;
            case SOFTPLUS -> UnaryFunctionType.SOFTPLUS;
            case SOFTSIGN -> UnaryFunctionType.SOFTSIGN;
            case RELU -> UnaryFunctionType.RELU;
            case RELU_COS -> UnaryFunctionType.RELU_COS;
            case RELU_SIN -> UnaryFunctionType.RELU_SIN;
            case ELU -> UnaryFunctionType.ELU;
            case SELU -> UnaryFunctionType.SELU;
            case GELU -> UnaryFunctionType.GELU;
            case SOFTMAX -> UnaryFunctionType.SOFTMAX;
            case GAUSSIAN -> UnaryFunctionType.GAUSSIAN;
            case SINACT -> UnaryFunctionType.SINACT;
            case MISH -> UnaryFunctionType.MISH;
            case LOGIT -> UnaryFunctionType.LOGIT;
            case CUSTOM -> UnaryFunctionType.CUSTOM;
        }, params);
    }

}
