/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.agent.Environment;
import core.reinforcement.agent.StateSynchronization;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.policy.updateablepolicy.UpdateableBasicPolicy;
import core.reinforcement.value.StateValueFunction;
import core.reinforcement.value.ValueFunction;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements Actor Critic algorithm.<br>
 *
 */
public class ActorCritic extends AbstractPolicyGradient {

    /**
     * Constructor for Actor Critic
     *
     * @param stateSynchronization    reference to state synchronization.
     * @param environment             reference to environment.
     * @param executablePolicyType    executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator  reference to value function estimator.
     * @param memory                  reference to memory.
     * @param params                  parameters
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException       throws exception if matrix operation fails.
     */
    public ActorCritic(StateSynchronization stateSynchronization, Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator, Memory memory, String params) throws DynamicParamException, MatrixException {
        super(stateSynchronization, environment, new UpdateableBasicPolicy(executablePolicyType, policyFunctionEstimator, memory, params), new StateValueFunction(valueFunctionEstimator, params), memory, params);
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
    public ActorCritic reference(boolean sharedPolicyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException {
        Memory newMemory = sharedMemory ? memory : memory.reference();
        ValueFunction newValueFunction = valueFunction.reference(sharedValueFunctionEstimator);
        Policy newPolicy = newValueFunction.getFunctionEstimator().isStateActionValueFunction() ? policy.reference(newValueFunction.getFunctionEstimator(), memory) : policy.reference(sharedPolicyFunctionEstimator, newMemory);
        return new ActorCritic(getStateSynchronization(), getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), newPolicy.getFunctionEstimator(), newValueFunction.getFunctionEstimator(), newMemory, getParams());
    }

}
