/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.State;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.TreeSet;

/**
 * Implements actionable policy.<br>
 *
 */
public class ActionablePolicy extends AbstractPolicy {

    /**
     * Constructor for actionable policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public ActionablePolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator);
    }

    /**
     * Constructor for actionable policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to function estimator.
     * @param params parameters for actionable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public ActionablePolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, String params) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
    }

    /**
     * Return true if policy is updateable otherwise false.
     *
     * @return true if policy is updateable otherwise false.
     */
    public boolean isUpdateablePolicy() {
        return false;
    }

    /**
     * Returns reference to policy.
     *
     * @return reference to policy.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference(FunctionEstimator valueFunctionEstimator) throws DynamicParamException, AgentException, MatrixException, IOException, ClassNotFoundException {
        return new ActionablePolicy(executablePolicy.getExecutablePolicyType(), getFunctionEstimator().reference(), params);
    }

    /**
     * Returns reference to policy.
     *
     * @param valueFunctionEstimator        reference to value function estimator.
     * @param sharedPolicyFunctionEstimator if true shared policy is used between value functions.
     * @param sharedMemory                  if true policy will use shared memory otherwise dedicated memory.
     * @return reference to policy.
     * @throws IOException            throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws AgentException         throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference(FunctionEstimator valueFunctionEstimator, boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws DynamicParamException, AgentException, MatrixException, IOException, ClassNotFoundException {
        return new ActionablePolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(sharedMemory), params);
    }

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
    public Policy reference(FunctionEstimator valueFunctionEstimator, boolean sharedPolicyFunctionEstimator, Memory memory) throws DynamicParamException, IOException, ClassNotFoundException, AgentException, MatrixException {
        return new ActionablePolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(memory), params);
    }

    /**
     * Ends episode
     *
     */
    public void endEpisode() {
    }

    /**
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @return true if all registered agents are ready to update.
     */
    public boolean readyToUpdate(Agent agent) {
        return true;
    }

    /**
     * Updates function estimator.
     *
     */
    public void updateFunctionEstimator() {
    }

    /**
     * Updates function estimator.
     *
     * @param sampledStates sampled states.
     */
    public void updateFunctionEstimator(TreeSet<State> sampledStates) {
    }

    /**
     * Resets function estimator.
     *
     */
    public void resetFunctionEstimator() {
    }

}
