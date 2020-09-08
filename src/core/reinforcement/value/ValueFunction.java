/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.NeuralNetworkException;
import core.reinforcement.Agent;
import core.reinforcement.AgentException;
import core.reinforcement.function.FunctionEstimator;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

/**
 * Interface that defines ValueFunction.
 *
 */
public interface ValueFunction {

    /**
     * Starts FunctionEstimator
     *
     * @throws NeuralNetworkException throws exception if starting of value function fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    void start() throws NeuralNetworkException, MatrixException;

    /**
     * Stops FunctionEstimator
     *
     */
    void stop();

    /**
     * Updates value function.
     *
     * @param agent agent.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    void update(Agent agent) throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException;

    /**
     * Returns FunctionEstimator.
     *
     * @return FunctionEstimator.
     */
    FunctionEstimator getFunctionEstimator();

}
