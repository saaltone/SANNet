/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
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
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void start() throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException;

    /**
     * Stops agent.
     *
     * @throws NeuralNetworkException throws exception is neural network is not started.
     */
    void stop() throws NeuralNetworkException;

    /**
     * Begins new step for agent.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    void newTimeStep() throws MatrixException, DynamicParamException, NeuralNetworkException, AgentException;

    /**
     * Starts episode.
     *
     */
    void startEpisode();

    /**
     * Ends episode.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
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
     * @throws AgentException throws exception if policy fails to choose valid action.
     */
    void act() throws NeuralNetworkException, MatrixException, AgentException;

    /**
     * Takes action defined by external agent.
     *
     * @param action action.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws AgentException throws exception if defined action is not available in current environment state.
     */
    void act(int action) throws NeuralNetworkException, MatrixException, AgentException;

    /**
     * Assigns immediate reward from environment in response to action agent executed.
     *
     *  @param reward immediate reward.
     */
    void respond(double reward);

    /**
     * Returns cumulative reward.
     *
     * @param isLearning if true returns cumulative reward during learning otherwise returns cumulative reward when not learning
     * @return cumulative reward.
     */
    double getCumulativeReward(boolean isLearning);

    /**
     * Returns moving average reward.
     *
     * @param isLearning if true returns moving average reward during learning otherwise returns moving average reward when not learning
     * @return moving average reward.
     */
    double getMovingAverageReward(boolean isLearning);

    /**
     * Resets cumulative and moving average reward metrics to zero.
     *
     */
    void resetRewardMetrics();

}
