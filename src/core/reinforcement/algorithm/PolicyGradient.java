/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.Environment;
import core.reinforcement.agent.StateSynchronization;
import core.reinforcement.policy.Policy;
import core.reinforcement.value.ValueFunction;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements generic Policy Gradient algorithm.<br>
 * Can be used to implement Actor Critic and REINFORCE algorithms.<br>
 *
 */
public class PolicyGradient extends AbstractPolicyGradient {

    /**
     * Constructor for Policy Gradient.
     *
     * @param stateSynchronization reference to state synchronization.
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to valueFunction.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public PolicyGradient(StateSynchronization stateSynchronization, Environment environment, Policy policy, ValueFunction valueFunction, String params) throws DynamicParamException {
        super(stateSynchronization, environment, policy, valueFunction, params);
    }

    /**
     * Returns reference to algorithm.
     *
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public PolicyGradient reference() throws DynamicParamException, MatrixException, AgentException, IOException, ClassNotFoundException {
        Policy newPolicy = policy.reference(null);
        ValueFunction newValueFunction = valueFunction.reference(false, newPolicy.getFunctionEstimator().getMemory());
        return new PolicyGradient(getStateSynchronization(), getEnvironment(), newPolicy, newValueFunction, getParams());
    }

    /**
     * Returns reference to algorithm.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public PolicyGradient reference(boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException {
        Policy newPolicy = policy.reference(null, sharedPolicyFunctionEstimator, sharedMemory);
        ValueFunction newValueFunction = valueFunction.reference(false, newPolicy.getFunctionEstimator().getMemory());
        return new PolicyGradient(getStateSynchronization(), getEnvironment(), newPolicy, newValueFunction, getParams());
    }

    /**
     * Returns reference to algorithm.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedValueFunctionEstimator if true shared value function estimator is used otherwise new policy function estimator is created.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public PolicyGradient reference(boolean sharedPolicyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException {
        Policy newPolicy = policy.reference(null, sharedPolicyFunctionEstimator, sharedMemory);
        ValueFunction newValueFunction = valueFunction.reference(sharedValueFunctionEstimator, newPolicy.getFunctionEstimator().getMemory());
        return new PolicyGradient(getStateSynchronization(), getEnvironment(), newPolicy, newValueFunction, getParams());
    }

}
