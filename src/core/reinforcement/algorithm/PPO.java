/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.agent.Environment;
import core.reinforcement.agent.StateSynchronization;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.updateablepolicy.UpdateableProximalPolicy;
import core.reinforcement.value.StateValueFunction;
import core.reinforcement.value.ValueFunction;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements Proximal Policy Optimization (PPO) algorithm.<br>
 *
 */
public class PPO extends AbstractPolicyGradient {

    /**
     * Constructor for Proximal Policy Optimization
     *
     * @param stateSynchronization    reference to state synchronization.
     * @param environment             reference to environment.
     * @param updateableProximalPolicy reference to updateable proximal policy.
     * @param stateValueFunction      reference to state value function.
     * @param memory                  reference to memory.
     * @param params                  parameters for agent.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public PPO(StateSynchronization stateSynchronization, Environment environment, UpdateableProximalPolicy updateableProximalPolicy, StateValueFunction stateValueFunction, Memory memory, String params) throws DynamicParamException {
        super(stateSynchronization, environment, updateableProximalPolicy, stateValueFunction, memory, params);
    }

    /**
     * Returns reference to algorithm.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedValueFunctionEstimator if true shared value function estimator is used otherwise new value function estimator is created.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public PPO reference(boolean sharedPolicyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException {
        Memory newMemory = getMemory(sharedMemory);
        ValueFunction newValueFunction = getValueFunction(sharedValueFunctionEstimator);
        Policy newPolicy = getPolicy(sharedPolicyFunctionEstimator, newValueFunction, newMemory);
        return new PPO(getStateSynchronization(), getEnvironment(), (UpdateableProximalPolicy) newPolicy, (StateValueFunction) newValueFunction, newMemory, getParams());
    }

}