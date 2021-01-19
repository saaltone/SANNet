/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.NeuralNetworkException;
import core.reinforcement.Agent;
import core.reinforcement.AgentException;
import core.reinforcement.Environment;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

/**
 * Interface for ActionablePolicy.
 *
 */
public interface Policy {

    /**
     * Starts ActionablePolicy.
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void start() throws NeuralNetworkException, MatrixException, DynamicParamException;

    /**
     * Stops ActionablePolicy.
     *
     */
    void stop();

    /**
     * Sets reference to environment.
     *
     * @param environment reference to environment.
     */
    void setEnvironment(Environment environment);

    /**
     * Returns reference to environment.
     *
     * @return reference to environment.
     */
    Environment getEnvironment();

    /**
     * Set flag if agent is in learning mode.
     *
     * @param isLearning if true agent is in learning mode.
     */
    void setLearning(boolean isLearning);

    /**
     * Return flag is policy is in learning mode.
     *
     * @return if true agent is in learning mode.
     */
    boolean isLearning();

    /**
     * Resets policy.
     *
     * @param forceReset forces to trigger reset.
     */
    void reset(boolean forceReset);

    /**
     * Updates policy.
     *
     */
    void increment();

    /**
     * Takes action defined by external agent.
     *
     * @param stateTransition state transition.
     * @param action action.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void act(StateTransition stateTransition, int action) throws MatrixException, NeuralNetworkException;

    /**
     * Takes action by applying defined executable policy,
     *
     * @param stateTransition state transition.
     * @param alwaysGreedy if true greedy action is always taken.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void act(StateTransition stateTransition, boolean alwaysGreedy) throws NeuralNetworkException, MatrixException;

    /**
     * Updates policy.
     *
     */
    void update();

    /**
     * Updates policy.
     *
     * @param agent agent.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    void update(Agent agent) throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException;

    /**
     * Returns FunctionEstimator.
     *
     * @return FunctionEstimator.
     */
    FunctionEstimator getFunctionEstimator();

}