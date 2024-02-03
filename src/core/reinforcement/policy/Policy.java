/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.agent.State;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.executablepolicy.ExecutablePolicy;
import utils.configurable.Configurable;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.TreeSet;

/**
 * Interface for Policy.
 *
 */
public interface Policy extends Configurable {

    /**
     * Return true if policy is updateable otherwise false.
     *
     * @return true if policy is updateable otherwise false.
     */
    boolean isUpdateablePolicy();

    /**
     * Returns reference to policy.
     *
     * @param valueFunctionEstimator reference to value function estimator.
     * @return reference to policy.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    Policy reference(FunctionEstimator valueFunctionEstimator) throws DynamicParamException, IOException, ClassNotFoundException, AgentException, MatrixException;

    /**
     * Returns reference to policy.
     *
     * @param valueFunctionEstimator        reference to value function estimator.
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedMemory                  if true policy will use shared memory otherwise dedicated memory.
     * @return reference to policy.
     * @throws IOException            throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws AgentException         throws exception if state action value function is applied to non-updateable policy.
     */
    Policy reference(FunctionEstimator valueFunctionEstimator, boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws DynamicParamException, IOException, ClassNotFoundException, AgentException, MatrixException;

    /**
     * Returns reference to policy.
     *
     * @param valueFunctionEstimator        reference to value function estimator.
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param memory                  if true policy will use shared memory otherwise dedicated memory.
     * @return reference to policy.
     * @throws IOException            throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws AgentException         throws exception if state action value function is applied to non-updateable policy.
     */
    Policy reference(FunctionEstimator valueFunctionEstimator, boolean sharedPolicyFunctionEstimator, Memory memory) throws DynamicParamException, IOException, ClassNotFoundException, AgentException, MatrixException;

    /**
     * Starts policy.
     *
     * @param agent agent.
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void start(Agent agent) throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException;

    /**
     * Stops policy.
     *
     * @throws NeuralNetworkException throws exception is neural network is not started.
     */
    void stop() throws NeuralNetworkException;

    /**
     * Returns executable policy.
     *
     * @return executable policy.
     */
    ExecutablePolicy getExecutablePolicy();

    /**
     * Sets flag if agent is in learning mode.
     *
     * @param isLearning if true agent is in learning mode.
     */
    void setLearning(boolean isLearning);

    /**
     * Return flag is agent is in learning mode.
     *
     * @return if true agent is in learning mode.
     */
    boolean isLearning();

    /**
     * Updates policy.
     *
     */
    void increment();

    /**
     * Takes action defined by external agent.
     *
     * @param state state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void act(State state) throws MatrixException, NeuralNetworkException;

    /**
     * Takes action defined by executable policy.
     *
     * @param state state.
     * @param alwaysGreedy if true greedy action is always taken.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void act(State state, boolean alwaysGreedy) throws NeuralNetworkException, MatrixException;

    /**
     * Ends episode
     *
     */
    void endEpisode();

    /**
     * Resets function estimator.
     *
     */
    void resetFunctionEstimator();

    /**
     * Returns reference to function estimator.
     *
     * @return reference to function estimator.
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
     * Updates function estimator.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    void updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException, IOException, ClassNotFoundException;

    /**
     * Updates function estimator.
     *
     * @param sampledStates sampled states.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    void updateFunctionEstimator(TreeSet<State> sampledStates) throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException, IOException, ClassNotFoundException;

}
