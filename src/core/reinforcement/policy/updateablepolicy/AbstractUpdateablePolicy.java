/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.State;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.AbstractPolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.util.TreeSet;

/**
 * Implements abstract updateable policy.<br>
 * Contains common functions fo updateable policies.<br>
 *
 */
public abstract class AbstractUpdateablePolicy extends AbstractPolicy {

    /**
     * Constructor for abstract updateable policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator    reference to function estimator.
     * @param memory               reference to memory.
     * @param params               parameters for abstract updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public AbstractUpdateablePolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, Memory memory, String params) throws DynamicParamException, MatrixException {
        super(executablePolicyType, functionEstimator, memory, params);
    }

    /**
     * Constructor for abstract updateable policy.
     *
     * @param executablePolicy  executable policy.
     * @param functionEstimator reference to function estimator.
     * @param memory            reference to memory.
     * @param params            parameters for AbstractExecutablePolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public AbstractUpdateablePolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator, Memory memory, String params) throws DynamicParamException, MatrixException {
        super(executablePolicy, functionEstimator, memory, params);
    }

    /**
     * Ends episode
     *
     */
    public void endEpisode() {
        getExecutablePolicy().endEpisode();
    }

    /**
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @throws AgentException throws exception if agent is not registered for function estimator.
     * @return true if all registered agents are ready to update.
     */
    public boolean readyToUpdate(Agent agent) throws AgentException {
        return getFunctionEstimator().readyToUpdate(agent);
    }

    /**
     * Returns policy gradient for state.
     *
     * @param state state.
     * @return policy gradient.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    protected abstract Matrix getPolicyGradient(State state) throws NeuralNetworkException, MatrixException, DynamicParamException;

    /**
     * Prepares function estimator update.
     *
     * @param sampledStates sampled states.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void prepareFunctionEstimator(TreeSet<State> sampledStates) throws NeuralNetworkException, MatrixException, DynamicParamException {
        for (State state : sampledStates) {
            getFunctionEstimator().storePolicyValues(state, getPolicyGradient(state));
        }
    }

    /**
     * Finishes function estimator update.
     *
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public void finishFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException {
        getFunctionEstimator().update();
    }

}
