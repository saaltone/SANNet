/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.StateTransition;
import utils.configurable.Configurable;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Interface that defines value function.<br>
 *
 */
public interface ValueFunction extends Configurable {

    /**
     * Returns reference to value function.
     *
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if soft Q alpha matrix is non-scalar matrix.
     */
    ValueFunction reference() throws DynamicParamException, MatrixException, IOException, ClassNotFoundException, AgentException;

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if soft Q alpha matrix is non-scalar matrix.
     */
    ValueFunction reference(boolean sharedValueFunctionEstimator, boolean sharedMemory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException, AgentException;

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param memory reference to memory.
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if soft Q alpha matrix is non-scalar matrix.
     */
    ValueFunction reference(boolean sharedValueFunctionEstimator, Memory memory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException, AgentException;

    /**
     * Starts function estimator
     *
     * @throws NeuralNetworkException throws exception if starting of value function fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void start() throws NeuralNetworkException, MatrixException, DynamicParamException;

    /**
     * Stops function estimator
     *
     */
    void stop();

    /**
     * Registers agent for function estimator.
     *
     * @param agent agent.
     */
    void registerAgent(Agent agent);

    /**
     * Updates value function.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void update() throws MatrixException, NeuralNetworkException;

    /**
     * Updates values for current episode.
     *
     * @param stateTransition state transition.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    void update(StateTransition stateTransition) throws NeuralNetworkException, MatrixException;

    /**
     * Resets function estimator.
     *
     */
    void resetFunctionEstimator();

    /**
     * Returns function estimator.
     *
     * @return function estimator.
     */
    FunctionEstimator getFunctionEstimator();

    /**
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @throws AgentException throws exception if agent is not registered for function estimator.
     * @return true if all registered agents are ready to update.
     */
    boolean readyToUpdate(Agent agent) throws AgentException;

    /**
     * Samples memory of function estimator.
     *
     */
    void sample();

    /**
     * Updates function estimator.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    void updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException;

}
