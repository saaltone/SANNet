/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.agent.Environment;
import core.reinforcement.agent.StateSynchronization;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.updateablepolicy.UpdateableQPolicy;
import core.reinforcement.value.QPolicyValueFunction;
import core.reinforcement.value.ValueFunction;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements Deep Deterministic Policy Gradient (DDPG) algorithm.<br>
 * Reference: <a href="https://towardsdatascience.com/deep-deterministic-policy-gradient-ddpg-theory-and-implementation-747a3010e82f">...</a> <br>
 *
 */
public class DDPG extends AbstractPolicyGradient {

    /**
     * Constructor for Deep Deterministic Policy Gradient
     *
     * @param stateSynchronization    reference to state synchronization.
     * @param environment             reference to environment.
     * @param updateableQPolicy       reference to updateable Q policy.
     * @param qPolicyValueFunction    reference to Q policy value function.
     * @param memory                  reference to memory.
     * @param params                  parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DDPG(StateSynchronization stateSynchronization, Environment environment, UpdateableQPolicy updateableQPolicy, QPolicyValueFunction qPolicyValueFunction, Memory memory, String params) throws DynamicParamException {
        super(stateSynchronization, environment, updateableQPolicy, qPolicyValueFunction, memory, params);
        updateableQPolicy.setQPolicyValueFunction(qPolicyValueFunction);
        qPolicyValueFunction.setUpdateableQPolicy(updateableQPolicy);
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
     */
    public DDPG reference(boolean sharedPolicyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException {
        Memory newMemory = sharedMemory ? memory : memory.reference();
        ValueFunction newValueFunction = valueFunction.reference(sharedValueFunctionEstimator);
        Policy newPolicy = newValueFunction.getFunctionEstimator().isStateActionValueFunction() ? policy.reference(newValueFunction.getFunctionEstimator(), memory) : policy.reference(sharedPolicyFunctionEstimator, newMemory);
        return new DDPG(getStateSynchronization(), getEnvironment(), (UpdateableQPolicy)newPolicy, (QPolicyValueFunction) newValueFunction, newMemory, getParams());
    }

}
