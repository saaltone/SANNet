/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.function;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.memory.Memory;
import core.reinforcement.agent.State;
import utils.configurable.Configurable;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.HashSet;
import java.util.TreeSet;

/**
 * Interface defining function estimator.<br>
 *
 */
public interface FunctionEstimator extends Configurable {

    /**
     * Returns reference to function estimator.
     *
     * @return reference to value function.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    FunctionEstimator reference() throws DynamicParamException, MatrixException, IOException, ClassNotFoundException;

    /**
     * Returns reference to function estimator.
     *
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to value function.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    FunctionEstimator reference(boolean sharedMemory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException;

    /**
     * Returns reference to function estimator.
     *
     * @param memory reference to memory.
     * @return reference to value function.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    FunctionEstimator reference(Memory memory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException;

    /**
     * Returns number of states for function estimator.
     *
     * @return number of states for function estimator.
     */
    int getNumberOfStates();

    /**
     * Returns number of actions for function estimator.
     *
     * @return number of actions for function estimator.
     */
    int getNumberOfActions();

    /**
     * Returns reference to memory of function estimator.
     *
     * @return reference to memory of function estimator.
     */
    Memory getMemory();

    /**
     * Starts function estimator.
     *
     * @throws NeuralNetworkException throws exception if starting of function estimator fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void start() throws NeuralNetworkException, MatrixException, DynamicParamException;

    /**
     * Stops function estimator.
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
     * Resets function estimator.
     *
     */
    void reset();

    /**
     * Samples memory of function estimator.
     *
     */
    void sample();

    /**
     * Returns sampled states.
     *
     * @return sampled states.
     */
    TreeSet<State> getSampledStates();

    /**
     * Adds new state into memory of function estimator.
     *
     * @param state state
     */
    void add(State state);

    /**
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @throws AgentException throws exception if agent is not registered for function estimator.
     * @return true if all registered agents are ready to update.
     */
    boolean readyToUpdate(Agent agent) throws AgentException;

    /**
     * Updates states in memory of function estimator.
     *
     * @param states states
     */
    void update(TreeSet<State> states);

    /**
     * If true value function is combined state action value function.
     *
     * @return true if value function is combined state action value function.
     */
    boolean isStateActionValueFunction();

    /**
     * Returns copy of function estimator.
     *
     * @return copy of function estimator.
     * @throws IOException throws exception if creation of function estimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of function estimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    FunctionEstimator copy() throws IOException, ClassNotFoundException, DynamicParamException, MatrixException;

    /**
     * Predicts target policy values corresponding to a state.
     *
     * @param state state.
     * @return policy values corresponding to a state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    Matrix predictTargetPolicyValues(State state) throws NeuralNetworkException, MatrixException;

    /**
     * Predicts policy values corresponding to a state.
     *
     * @param state state.
     * @return policy values corresponding to a state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    Matrix predictPolicyValues(State state) throws NeuralNetworkException, MatrixException;

    /**
     * Predicts state action values corresponding to a state.
     *
     * @param state state.
     * @return state action values corresponding to a state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix predictStateActionValues(State state) throws NeuralNetworkException, MatrixException;

    /**
     * Predicts target state action values corresponding to a state.
     *
     * @param state state.
     * @return state action values corresponding to a state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix predictTargetStateActionValues(State state) throws NeuralNetworkException, MatrixException;

    /**
     * Stores policy state values pair.
     *
     * @param state state.
     * @param values values.
     */
    void storePolicyValues(State state, Matrix values);

    /**
     * Stores state action values pair.
     *
     * @param state state.
     * @param values values.
     */
    void storeStateActionValues(State state, Matrix values);

    /**
     * Updates (trains) function estimator.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void update() throws NeuralNetworkException, MatrixException, DynamicParamException;

    /**
     * Aborts function estimator update.
     *
     */
    void abortUpdate();

    /**
     * Appends from function estimator.
     *
     * @param functionEstimator function estimator.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void append(FunctionEstimator functionEstimator) throws MatrixException;

    /**
     * Returns max value of state given available actions.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return max value of state.
     */
    double max(Matrix stateValues, HashSet<Integer> availableActions);

    /**
     * Returns action with maximum state value given available actions.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return action with maximum state value.
     */
    int argmax(Matrix stateValues, HashSet<Integer> availableActions);

    /**
     * Sets if importance sampling weights are applied.
     *
     * @param applyImportanceSamplingWeights if true importance sampling weights are applied otherwise not.
     */
    void setEnableImportanceSamplingWeights(boolean applyImportanceSamplingWeights);

}
