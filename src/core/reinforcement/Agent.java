/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

import core.NeuralNetworkException;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

/**
 * Interface for agent.
 *
 */
public interface Agent {

    /**
     * Starts agent.
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    void start() throws NeuralNetworkException, MatrixException;

    /**
     * Stops agent.
     *
     */
    void stop();

    /**
     * Starts new episode. Resets sequence by default.
     *
     */
    void newEpisode();

    /**
     * Begins new episode step for agent.
     *
     */
    void newStep();

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
     * Takes action as defined by agent's policy.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void act() throws NeuralNetworkException, MatrixException, DynamicParamException;

    /**
     * Takes action per given policy.
     *
     * @param alwaysGreedy if true greedy action is always taken.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void act(boolean alwaysGreedy) throws NeuralNetworkException, MatrixException, DynamicParamException;

    /**
     * Response from environment after agent commits action.
     *
     * @param reward immediate reward.
     * @param finalState if true state is final otherwise false.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void respond(double reward, boolean finalState) throws NeuralNetworkException, MatrixException, DynamicParamException;

}
