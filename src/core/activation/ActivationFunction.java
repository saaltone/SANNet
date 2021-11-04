/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.activation;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunction;
import utils.matrix.UnaryFunctionType;

import java.io.Serial;

/**
 * Defines activation function class for neural network and uses implementation of UnaryFunction.<br>
 * Provides calculation for both function and it's derivative.<br>
 * <br>
 * Following functions are supported:
 *     LINEAR,
 *     SIGMOID,
 *     SWISH,
 *     HARDSIGMOID,
 *     BIPOLARSIGMOID,
 *     TANH,
 *     TANHSIG,
 *     TANHAPPR,
 *     HARDTANH,
 *     EXP,
 *     SOFTPLUS,
 *     SOFTSIGN,
 *     RELU,
 *     RELU_COS,
 *     RELU_SIN,
 *     ELU,
 *     SELU,
 *     GELU,
 *     SOFTMAX,
 *     GUMBEL_SOFTMAX,
 *     GAUSSIAN,
 *     SINACT,
 *     LOGIT,
 *     CUSTOM
 */
public class ActivationFunction extends UnaryFunction {

    @Serial
    private static final long serialVersionUID = 4302816456291628148L;

    /**
     * List of supported activation functions.
     *
     */
    private final UnaryFunctionType[] activationFunctions = new UnaryFunctionType[] {
            UnaryFunctionType.LINEAR,
            UnaryFunctionType.SIGMOID,
            UnaryFunctionType.SWISH,
            UnaryFunctionType.HARDSIGMOID,
            UnaryFunctionType.BIPOLARSIGMOID,
            UnaryFunctionType.TANH,
            UnaryFunctionType.TANHSIG,
            UnaryFunctionType.TANHAPPR,
            UnaryFunctionType.HARDTANH,
            UnaryFunctionType.EXP,
            UnaryFunctionType.SOFTPLUS,
            UnaryFunctionType.SOFTSIGN,
            UnaryFunctionType.RELU,
            UnaryFunctionType.RELU_COS,
            UnaryFunctionType.RELU_SIN,
            UnaryFunctionType.ELU,
            UnaryFunctionType.SELU,
            UnaryFunctionType.GELU,
            UnaryFunctionType.SOFTMAX,
            UnaryFunctionType.GUMBEL_SOFTMAX,
            UnaryFunctionType.GAUSSIAN,
            UnaryFunctionType.SINACT,
            UnaryFunctionType.LOGIT,
            UnaryFunctionType.CUSTOM
    };

    /**
     * Constructor for activation function.
     *
     * @param unaryFunctionType type of activation function to be used.
     * @throws NeuralNetworkException throws exception if function is not available as activation function.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ActivationFunction(UnaryFunctionType unaryFunctionType) throws NeuralNetworkException, MatrixException, DynamicParamException {
        super(unaryFunctionType);
        for (UnaryFunctionType activationFunctionType : activationFunctions) {
            if (activationFunctionType == unaryFunctionType) return;
        }
        throw new NeuralNetworkException("No such activation function available: " + unaryFunctionType);
    }

    /**
     * Constructor for activation function.<br>
     *
     * @param unaryFunctionType type of activation function to be used.
     * @param params parameters used for activation function.
     * @throws NeuralNetworkException throws exception if function is not available as activation function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public ActivationFunction(UnaryFunctionType unaryFunctionType, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super(unaryFunctionType, params);
        for (UnaryFunctionType activationFunctionType : activationFunctions) {
            if (activationFunctionType == unaryFunctionType) return;
        }
        throw new NeuralNetworkException("No such activation function available: " + unaryFunctionType);
    }

}
