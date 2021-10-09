/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.Environment;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.policy.executablepolicy.ExecutablePolicy;
import core.reinforcement.value.ValueFunction;
import utils.Configurable;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

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
     * @return reference to policy.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    Policy reference() throws DynamicParamException, IOException, ClassNotFoundException, AgentException;

    /**
     * Returns reference to policy.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedMemory if true policy will use shared memory otherwise dedicated memory.
     * @return reference to policy.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if optimizer is of an unknown type.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    Policy reference(boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws DynamicParamException, IOException, ClassNotFoundException, NeuralNetworkException, AgentException;

    /**
     * Return true is function is state action value function.
     *
     * @return true is function is state action value function.
     */
    boolean isStateActionValueFunction();

    /**
     * Starts Policy.
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void start() throws NeuralNetworkException, MatrixException, DynamicParamException;

    /**
     * Stops Policy.
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
     * Returns executable policy.
     *
     * @return executable policy.
     */
    ExecutablePolicy getExecutablePolicy();

    /**
     * Sets value function for policy.
     *
     * @param valueFunction value function.
     */
    void setValueFunction(ValueFunction valueFunction);

    /**
     * Sets flag if agent is in learning mode.
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
     * Takes action defined by executable policy.
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
     * Resets FunctionEstimator.
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
     * Updates FunctionEstimator.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    void updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException;

}
