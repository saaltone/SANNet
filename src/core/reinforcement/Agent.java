/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement;

import core.NeuralNetworkException;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

/**
 * Interface defining agent.
 *
 */
public interface Agent {

    /**
     * Starts agent.
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void start() throws NeuralNetworkException, MatrixException;

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
     * Begins new (episode) step for agent.
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
     * Assigns immediate reward from environment in response to action agent executed.
     *
     *  @param reward immediate reward.
     */
    void respond(double reward);

}
