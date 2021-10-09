/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.StateTransition;
import utils.Configurable;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.TreeSet;

/**
 * Interface that defines ValueFunction.<br>
 *
 */
public interface ValueFunction extends Configurable {

    /**
     * Returns reference to value function.
     *
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if optimizer is of an unknown type.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if soft Q alpha matrix is non-scalar matrix.
     */
    ValueFunction reference() throws DynamicParamException, MatrixException, NeuralNetworkException, IOException, ClassNotFoundException, AgentException;

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws NeuralNetworkException throws exception if optimizer is of an unknown type.
     * @throws AgentException throws exception if soft Q alpha matrix is non-scalar matrix.
     */
    ValueFunction reference(boolean sharedValueFunctionEstimator, boolean sharedMemory) throws DynamicParamException, MatrixException, NeuralNetworkException, IOException, ClassNotFoundException, AgentException;

    /**
     * Starts FunctionEstimator
     *
     * @throws NeuralNetworkException throws exception if starting of value function fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void start() throws NeuralNetworkException, MatrixException, DynamicParamException;

    /**
     * Stops FunctionEstimator
     *
     */
    void stop();

    /**
     * Registers agent for FunctionEstimator.
     *
     * @param agent agent.
     */
    void registerAgent(Agent agent);

    /**
     * Return true is function is state action value function.
     *
     * @return true is function is state action value function.
     */
    boolean isStateActionValueFunction();

    /**
     * Returns value for state.
     *
     * @param stateTransition state transition.
     * @return value for state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
   double getValue(StateTransition stateTransition) throws MatrixException, NeuralNetworkException;

    /**
     * Returns target value based on next state.
     *
     * @param nextStateTransition next state transition.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    double getTargetValue(StateTransition nextStateTransition) throws NeuralNetworkException, MatrixException;

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
     * Resets FunctionEstimator.
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
     * Updates state transitions in memory of FunctionEstimator.
     *
     * @param stateTransitions state transitions
     */
    void updateFunctionEstimatorMemory(TreeSet<StateTransition> stateTransitions);

    /**
     * Samples memory of FunctionEstimator.
     *
     */
    void sample();

    /**
     * Returns sampled state transitions.
     *
     * @return sampled state transitions.
     */
    TreeSet<StateTransition> getSampledStateTransitions();

    /**
     * Updates FunctionEstimator.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    void updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException;

}
