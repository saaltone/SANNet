/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.function;

import core.NeuralNetworkException;
import core.reinforcement.Agent;
import core.reinforcement.AgentException;
import core.reinforcement.memory.StateTransition;
import utils.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.TreeSet;

/**
 * Interface defining FunctionEstimator.
 *
 */
public interface FunctionEstimator {

    /**
     * Returns number of actions for FunctionEstimator.
     *
     * @return number of actions for FunctionEstimator.
     */
    int getNumberOfActions();

    /**
     * Starts FunctionEstimator
     *
     * @throws NeuralNetworkException throws exception if starting of FunctionEstimator fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    void start() throws NeuralNetworkException, MatrixException;

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
     * Samples memory of FunctionEstimator.
     *
     */
    void sample();

    /**
     * Returns true if sample set is empty after sampling.
     *
     * @return true if sample set is empty after sampling.
     */
    boolean sampledSetEmpty();

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
     * Updated state transition in memory of FunctionEstimator.
     *
     * @param stateTransition state transition
     */
    void update(StateTransition stateTransition);

    /**
     * Updated state transitions in memory of FunctionEstimator.
     *
     * @param stateTransitions state transitions
     */
    void update(TreeSet<StateTransition> stateTransitions);

    /**
     * If true value function is combined state action value function.
     *
     * @return true if value function is combined state action value function.
     */
    boolean isStateActionValue();

    /**
     * Returns copy of FunctionEstimator.
     *
     * @return copy of FunctionEstimator.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    FunctionEstimator copy() throws IOException, ClassNotFoundException, DynamicParamException, MatrixException;

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
     * @param agent agent.
     * @param stateTransition state transition.
     * @param values values.
     * @throws AgentException throws exception if agent tries to store values outside ongoing update cycle.
     */
    void store(Agent agent, StateTransition stateTransition, Matrix values) throws AgentException;

    /**
     * Updates (trains) FunctionEstimator.
     *
     * @param agent agent
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if agent is not registered for ongoing update cycle.
     */
    void update(Agent agent) throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException;

    /**
     * Appends parameters to this FunctionEstimator from another FunctionEstimator.
     *
     * @param functionEstimator FunctionEstimator used to update current FunctionEstimator.
     * @param fullUpdate if true full update is done.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    void append(FunctionEstimator functionEstimator, boolean fullUpdate) throws MatrixException, AgentException;

}
