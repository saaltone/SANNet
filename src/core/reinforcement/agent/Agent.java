/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.agent;

import core.network.NeuralNetworkException;
import core.reinforcement.policy.Policy;
import core.reinforcement.value.ValueFunction;
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
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    void newStep() throws MatrixException, DynamicParamException, NeuralNetworkException, AgentException, IOException, ClassNotFoundException;

    /**
     * Ends episode.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    void endEpisode() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException, IOException, ClassNotFoundException;

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

    /**
     * Returns reference to algorithm.
     *
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws NeuralNetworkException throws exception if starting of function estimator fails.
     */
    Agent reference() throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException, NeuralNetworkException;

    /**
     * Appends parameters to this agent from another agent.
     *
     * @param agent agent used to update current agent.
     * @param tau tau which controls contribution of other agent.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    void append(Agent agent, double tau) throws MatrixException, AgentException, NeuralNetworkException, IOException, DynamicParamException, ClassNotFoundException;

    /**
     * Returns policy of agent.
     *
     * @return policy of agent.
     */
    Policy getPolicy();

    /**
     * Returns value function of agent.
     *
     * @return value function of agent.
     */
    ValueFunction getValueFunction();

}
