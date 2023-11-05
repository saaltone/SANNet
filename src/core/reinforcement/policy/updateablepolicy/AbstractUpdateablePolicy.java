/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.State;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.AbstractPolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.IOException;
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
     * @param functionEstimator reference to function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public AbstractUpdateablePolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator);
    }

    /**
     * Constructor for abstract updateable policy.
     *
     * @param executablePolicy executable policy.
     * @param functionEstimator reference to function estimator.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractUpdateablePolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator) throws AgentException, DynamicParamException {
        super(executablePolicy, functionEstimator);
    }

    /**
     * Constructor for abstract updateable policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to function estimator.
     * @param params parameters for abstract updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public AbstractUpdateablePolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, String params) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator, params);
    }

    /**
     * Constructor for abstract updateable policy.
     *
     * @param executablePolicy executable policy.
     * @param functionEstimator reference to function estimator.
     * @param params parameters for AbstractExecutablePolicy.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractUpdateablePolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator, String params) throws AgentException, DynamicParamException {
        super(executablePolicy, functionEstimator, params);
    }

    /**
     * Return true if policy is updateable otherwise false.
     *
     * @return true if policy is updateable otherwise false.
     */
    public boolean isUpdateablePolicy() {
        return true;
    }

    /**
     * Ends episode
     *
     */
    public void endEpisode() {
        getExecutablePolicy().endEpisode();
    }

    /**
     * Resets function estimator.
     *
     */
    public void resetFunctionEstimator() {
        getFunctionEstimator().reset();
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
     * Updates function estimator.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException, IOException, ClassNotFoundException {
        TreeSet<State> sampledStates = getFunctionEstimator().getSampledStates();
        if (sampledStates == null || sampledStates.isEmpty()) {
            getFunctionEstimator().abortUpdate();
            return;
        }

        for (State state : sampledStates) getFunctionEstimator().storePolicyValues(state, getPolicyValues(state));

        postProcess();

        getFunctionEstimator().update();
    }

    /**
     * Returns policy values.
     *
     * @param state state.
     * @return policy values.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    private Matrix getPolicyValues(State state) throws MatrixException, NeuralNetworkException {
        Matrix policyValues = new DMatrix(getFunctionEstimator().getNumberOfActions(), 1, 1);
        policyValues.setValue(state.action, 0, 0, getPolicyValue(state));
        return policyValues;
    }

    /**
     * Returns policy value for state.
     *
     * @param state state.
     * @return policy gradient value for sample.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract double getPolicyValue(State state) throws NeuralNetworkException, MatrixException;

    /**
     * Postprocesses policy gradient update.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void postProcess() throws MatrixException {
    }

}
