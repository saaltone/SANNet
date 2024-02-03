/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.agent.Environment;
import core.reinforcement.agent.StateSynchronization;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.updateablepolicy.UpdateableSoftQPolicy;
import core.reinforcement.value.SoftQValueFunction;
import core.reinforcement.value.ValueFunction;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements discrete Soft Actor Critic (SAC) algorithm.<br>
 *
 */
public class SoftActorCriticDiscrete extends AbstractPolicyGradient {

    /**
     * Constructor for discrete Soft Actor Critic
     *
     * @param stateSynchronization    reference to state synchronization.
     * @param environment             reference to environment.
     * @param updateableSoftQPolicy   reference to updateable soft Q policy.
     * @param softQValueFunction      reference to soft Q value function.
     * @param memory                  reference to memory.
     * @param params                  parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public SoftActorCriticDiscrete(StateSynchronization stateSynchronization, Environment environment, UpdateableSoftQPolicy updateableSoftQPolicy, SoftQValueFunction softQValueFunction, Memory memory, String params) throws DynamicParamException {
        super(stateSynchronization, environment, updateableSoftQPolicy, softQValueFunction, memory, params);
        updateableSoftQPolicy.setSoftQValueFunction(softQValueFunction);
        softQValueFunction.setUpdateableSoftQPolicy(updateableSoftQPolicy);
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
    public SoftActorCriticDiscrete reference(boolean sharedPolicyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException {
        Memory newMemory = sharedMemory ? memory : memory.reference();
        ValueFunction newValueFunction = valueFunction.reference(sharedValueFunctionEstimator);
        Policy newPolicy = newValueFunction.getFunctionEstimator().isStateActionValueFunction() ? policy.reference(newValueFunction.getFunctionEstimator(), memory) : policy.reference(sharedPolicyFunctionEstimator, newMemory);
        return new SoftActorCriticDiscrete(getStateSynchronization(), getEnvironment(), (UpdateableSoftQPolicy) newPolicy, (SoftQValueFunction) newValueFunction, newMemory, getParams());
    }

}
