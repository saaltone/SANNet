/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.function;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.memory.StateTransition;
import utils.configurable.Configurable;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.HashSet;
import java.util.TreeSet;

/**
 * Interface defining FunctionEstimator.<br>
 *
 */
public interface FunctionEstimator extends Configurable {

    /**
     * Returns reference to function estimator.
     *
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    FunctionEstimator reference() throws DynamicParamException;

    /**
     * Returns reference to function estimator.
     *
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    FunctionEstimator reference(boolean sharedMemory) throws DynamicParamException;

    /**
     * Returns number of states for FunctionEstimator.
     *
     * @return number of states for FunctionEstimator.
     */
    int getNumberOfStates();

    /**
     * Returns number of actions for FunctionEstimator.
     *
     * @return number of actions for FunctionEstimator.
     */
    int getNumberOfActions();

    /**
     * Starts FunctionEstimator.
     *
     * @throws NeuralNetworkException throws exception if starting of FunctionEstimator fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void start() throws NeuralNetworkException, MatrixException, DynamicParamException;

    /**
     * Stops FunctionEstimator.
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
     * Resets FunctionEstimator.
     *
     */
    void reset();

    /**
     * Reinitializes FunctionEstimator.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void reinitialize() throws MatrixException;

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
     * Adds new state transition into memory of FunctionEstimator.
     *
     * @param stateTransition state transition
     */
    void add(StateTransition stateTransition);

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
    void update(TreeSet<StateTransition> stateTransitions);

    /**
     * If true value function is combined state action value function.
     *
     * @return true if value function is combined state action value function.
     */
    boolean isStateActionValueFunction();

    /**
     * Returns copy of FunctionEstimator.
     *
     * @return copy of FunctionEstimator.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    FunctionEstimator copy() throws IOException, ClassNotFoundException, DynamicParamException;

    /**
     * Predicts state values corresponding to a state.
     *
     * @param state state.
     * @return state values corresponding to a state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix predict(Matrix state) throws NeuralNetworkException, MatrixException;

    /**
     * Stores state transition values pair.
     *
     * @param stateTransition state transition.
     * @param values values.
     */
    void store(StateTransition stateTransition, Matrix values);

    /**
     * Updates (trains) FunctionEstimator.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    void update() throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException;

    /**
     * Appends parameters to this FunctionEstimator from another FunctionEstimator.
     *
     * @param functionEstimator FunctionEstimator used to update current FunctionEstimator.
     * @param fullUpdate if true full update is done.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    void append(FunctionEstimator functionEstimator, boolean fullUpdate) throws MatrixException, AgentException;

    /**
     * Returns min value of state.
     *
     * @param stateValues state values.
     * @return min value of state.
     */
    double min(Matrix stateValues);

    /**
     * Returns min value of state given available actions.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return min value of state.
     */
    double min(Matrix stateValues, HashSet<Integer> availableActions);

    /**
     * Returns action with minimum state value.
     *
     * @param stateValues state values.
     * @return action with minimum state value.
     */
    int argmin(Matrix stateValues);

    /**
     * Returns action with minimum state value given available actions.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return action with minimum state value.
     */
    int argmin(Matrix stateValues, HashSet<Integer> availableActions);

    /**
     * Returns max value of state.
     *
     * @param stateValues state values.
     * @return max value of state.
     */
    double max(Matrix stateValues);

    /**
     * Returns max value of state given available actions.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return max value of state.
     */
    double max(Matrix stateValues, HashSet<Integer> availableActions);

    /**
     * Returns action with maximum state value.
     *
     * @param stateValues state values.
     * @return action with maximum state value.
     */
    int argmax(Matrix stateValues);

    /**
     * Returns action with maximum state value given available actions.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return action with maximum state value.
     */
    int argmax(Matrix stateValues, HashSet<Integer> availableActions);

    /**
     * Sets target function estimator.
     *
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void setTargetFunctionEstimator() throws ClassNotFoundException, DynamicParamException, IOException;

    /**
     * Returns target function estimator.
     *
     * @return target function estimator.
     */
    FunctionEstimator getTargetFunctionEstimator();

}
