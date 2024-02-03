/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.agent.State;
import utils.configurable.Configurable;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.TreeSet;

/**
 * Interface that defines value function.<br>
 *
 */
public interface ValueFunction extends Configurable {

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @return reference to value function.
     * @throws IOException            throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws MatrixException        throws exception if neural network has less output than actions.
     */
    ValueFunction reference(boolean sharedValueFunctionEstimator) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException;

    /**
     * Starts function estimator
     *
     * @param agent agent.
     * @throws NeuralNetworkException throws exception if starting of value function fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void start(Agent agent) throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException;

    /**
     * Stops function estimator
     *
     * @throws NeuralNetworkException throws exception is neural network is not started.
     */
    void stop() throws NeuralNetworkException;

    /**
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @throws AgentException throws exception if agent is not registered for function estimator.
     * @return true if all registered agents are ready to update.
     */
    boolean readyToUpdate(Agent agent) throws AgentException;

    /**
     * Updates value function.
     *
     * @param sampledStates sampled states.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    void update(TreeSet<State> sampledStates) throws MatrixException, NeuralNetworkException, DynamicParamException;

    /**
     * Returns function estimator.
     *
     * @return function estimator.
     */
    FunctionEstimator getFunctionEstimator();

    /**
     * Updates function estimator.
     *
     * @param sampledStates sampled states.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    void updateFunctionEstimator(TreeSet<State> sampledStates) throws NeuralNetworkException, MatrixException, DynamicParamException;

}
