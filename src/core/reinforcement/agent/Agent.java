/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.agent;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Interface defining agent.<br>
 *
 */
public interface Agent {

    /**
     * Starts agent.
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void start() throws NeuralNetworkException, MatrixException, DynamicParamException;

    /**
     * Stops agent.
     *
     */
    void stop();

    /**
     * Starts new episode for episodic environments.
     *
     */
    void newEpisode();

    /**
     * Begins new step for agent.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if memory instances of value and policy function are not equal.
     */
    void newStep() throws MatrixException, DynamicParamException, NeuralNetworkException, AgentException;

    /**
     * Ends episode.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update of estimator fails.
     */
    void endEpisode() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException;

    /**
     * Disables learning.
     *
     */
    void disableLearning();

    /**
     * Enables learning.
     *
     */
    void enableLearning();

    /**
     * Takes action per defined agent policy.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void act() throws NeuralNetworkException, MatrixException;

    /**
     * Takes action per defined agent policy.
     *
     * @param alwaysGreedy if true greedy action is always taken.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void act(boolean alwaysGreedy) throws NeuralNetworkException, MatrixException;

    /**
     * Takes action defined by external agent.
     *
     * @param action action.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void act(int action) throws NeuralNetworkException, MatrixException;

    /**
     * Assigns immediate reward from environment in response to action agent executed.
     *
     *  @param reward immediate reward.
     */
    void respond(double reward);

    /**
     * Resets policy.
     *
     */
    void resetPolicy();

    /**
     * Returns reference to algorithm.
     *
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws NeuralNetworkException throws exception if optimizer is of an unknown type.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    Agent reference() throws MatrixException, NeuralNetworkException, IOException, DynamicParamException, ClassNotFoundException, AgentException;

}
