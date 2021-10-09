/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.DynamicParamException;

/**
 * Class that defines ActionablePolicy.<br>
 *
 */
public class ActionablePolicy extends AbstractPolicy {

    /**
     * Constructor for ActionablePolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public ActionablePolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator);
    }

    /**
     * Constructor for ActionablePolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for ActionablePolicy.
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
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference() throws DynamicParamException, AgentException {
        return new ActionablePolicy(executablePolicy.getExecutablePolicyType(), functionEstimator, params);
    }

    /**
     * Returns reference to policy.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy is used between value functions.
     * @param sharedMemory if true policy will use shared memory otherwise dedicated memory.
     * @return reference to policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Policy reference(boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws DynamicParamException, AgentException {
        return new ActionablePolicy(executablePolicy.getExecutablePolicyType(), functionEstimator, params);
    }

    /**
     * Updates policy.
     *
     */
    public void update() {
    }

    /**
     * Resets FunctionEstimator.
     *
     */
    public void resetFunctionEstimator() {
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
     * Updates policy.
     *
     */
    public void updateFunctionEstimator() {
    }

}
