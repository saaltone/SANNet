/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.activation;

import core.NeuralNetworkException;
import utils.*;
import utils.matrix.UnaryFunction;
import utils.matrix.UnaryFunctionType;

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
 *     TANSIG,
 *     TANHAPPR,
 *     HARDTANH,
 *     EXPONENTIAL,
 *     SOFTPLUS,
 *     SOFTSIGN,
 *     RELU,
 *     ELU,
 *     SELU,
 *     GELU,
 *     SOFTMAX,
 *     GAUSSIAN,
 *     SIN
 */
public class ActivationFunction extends UnaryFunction {

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
            UnaryFunctionType.TANSIG,
            UnaryFunctionType.TANHAPPR,
            UnaryFunctionType.HARDTANH,
            UnaryFunctionType.EXP,
            UnaryFunctionType.SOFTPLUS,
            UnaryFunctionType.SOFTSIGN,
            UnaryFunctionType.RELU,
            UnaryFunctionType.ELU,
            UnaryFunctionType.SELU,
            UnaryFunctionType.GELU,
            UnaryFunctionType.SOFTMAX,
            UnaryFunctionType.GAUSSIAN,
            UnaryFunctionType.SIN
    };

    /**
     * Constructor for activation function.
     *
     * @param unaryFunctionType type of activation function to be used.
     * @throws NeuralNetworkException throws exception if function is not available as activation function.
     */
    public ActivationFunction(UnaryFunctionType unaryFunctionType) throws NeuralNetworkException {
        super(unaryFunctionType);
        boolean found = false;
        for (UnaryFunctionType fType : activationFunctions) {
            if (fType == unaryFunctionType) {
                found = true;
                break;
            }
        }
        if (!found) throw new NeuralNetworkException("No such activation function available.");
    }

    /**
     * Constructor for activation function.<br>
     *
     * @param unaryFunctionType type of activation function to be used.
     * @param params parameters used for activation function.
     * @throws NeuralNetworkException throws exception if function is not available as activation function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ActivationFunction(UnaryFunctionType unaryFunctionType, String params) throws NeuralNetworkException, DynamicParamException {
        super(unaryFunctionType, params);
        boolean found = false;
        for (UnaryFunctionType fType : activationFunctions) {
            if (fType == unaryFunctionType) {
                found = true;
                break;
            }
        }
        if (!found) throw new NeuralNetworkException("No such activation function available.");
    }

}
