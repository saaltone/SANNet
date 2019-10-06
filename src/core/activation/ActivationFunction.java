/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.activation;

import core.NeuralNetworkException;
import utils.*;

/**
 * Defines activation function class for neural network and uses implementation of UniFunction.<br>
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
public class ActivationFunction extends UniFunction {

    private static final long serialVersionUID = 4302816456291628148L;

    /**
     * List of supported activation functions.
     *
     */
    private final UniFunctionType[] activationFunctions = new UniFunctionType[] {
            UniFunctionType.LINEAR,
            UniFunctionType.SIGMOID,
            UniFunctionType.SWISH,
            UniFunctionType.HARDSIGMOID,
            UniFunctionType.BIPOLARSIGMOID,
            UniFunctionType.TANH,
            UniFunctionType.TANSIG,
            UniFunctionType.TANHAPPR,
            UniFunctionType.HARDTANH,
            UniFunctionType.EXP,
            UniFunctionType.SOFTPLUS,
            UniFunctionType.SOFTSIGN,
            UniFunctionType.RELU,
            UniFunctionType.ELU,
            UniFunctionType.SELU,
            UniFunctionType.GELU,
            UniFunctionType.SOFTMAX,
            UniFunctionType.GAUSSIAN,
            UniFunctionType.SIN
    };

    /**
     * Constructor for activation function.
     *
     * @param uniFunctionType type of activation function to be used.
     * @throws NeuralNetworkException throws exception if function is not available as activation function.
     */
    public ActivationFunction(UniFunctionType uniFunctionType) throws NeuralNetworkException {
        super(uniFunctionType);
        boolean found = false;
        for (UniFunctionType fType : activationFunctions) {
            if (fType == uniFunctionType) {
                found = true;
                break;
            }
        }
        if (!found) throw new NeuralNetworkException("No such activation function available.");
    }

    /**
     * Constructor for activation function.<br>
     *
     * @param uniFunctionType type of activation function to be used.
     * @param params parameters used for activation function.
     * @throws NeuralNetworkException throws exception if function is not available as activation function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ActivationFunction(UniFunctionType uniFunctionType, String params) throws NeuralNetworkException, DynamicParamException {
        super(uniFunctionType, params);
        boolean found = false;
        for (UniFunctionType fType : activationFunctions) {
            if (fType == uniFunctionType) {
                found = true;
                break;
            }
        }
        if (!found) throw new NeuralNetworkException("No such activation function available.");
    }

}
