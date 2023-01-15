/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.agent;

import core.network.NeuralNetwork;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

/**
 * Interface for agent function estimator creation.
 *
 */
public interface AgentFunctionEstimator {

    /**
     * Build neural network for agent (separate functions for state action value function estimator).
     *
     * @param inputSize input size of neural network (number of states)
     * @param outputSize output size of neural network (number of actions and their values).
     * @param policyGradient if true neural network is of type policy gradient.
     * @param applyDueling if true applied dueling layer to non policy gradient network otherwise not.
     * @return built neural network
     * @throws DynamicParamException throws exception if setting of dynamic parameters fails.
     * @throws NeuralNetworkException throws exception if building of neural network fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize, boolean policyGradient, boolean applyDueling) throws DynamicParamException, NeuralNetworkException, MatrixException;

    /**
     * Build neural network for agent (shared function for state action value function estimator).
     *
     * @param inputSize input size of neural network (number of states)
     * @param outputSize output size of neural network (number of actions and their values).
     * @return built neural network
     * @throws DynamicParamException throws exception if setting of dynamic parameters fails.
     * @throws NeuralNetworkException throws exception if building of neural network fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize) throws DynamicParamException, NeuralNetworkException, MatrixException;

}
