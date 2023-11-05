/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.Environment;
import core.reinforcement.agent.StateSynchronization;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.policy.updateablepolicy.UpdateableSoftQPolicy;
import core.reinforcement.value.SoftQValueFunctionEstimator;
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
     * @param stateSynchronization reference to state synchronization.
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param softQValueFunctionEstimator reference to soft Q value function estimator.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public SoftActorCriticDiscrete(StateSynchronization stateSynchronization, Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, SoftQValueFunctionEstimator softQValueFunctionEstimator, String params) throws DynamicParamException, AgentException {
        super(stateSynchronization, environment, new UpdateableSoftQPolicy(executablePolicyType, policyFunctionEstimator, softQValueFunctionEstimator), softQValueFunctionEstimator, params);
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
    public SoftActorCriticDiscrete reference() throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException {
        Policy newPolicy = policy.reference(null);
        ValueFunction newValueFunction = valueFunction.reference(newPolicy.getFunctionEstimator(), false, newPolicy.getFunctionEstimator().getMemory());
        return new SoftActorCriticDiscrete(getStateSynchronization(), getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), newPolicy.getFunctionEstimator(), (SoftQValueFunctionEstimator) newValueFunction, getParams());
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
    public SoftActorCriticDiscrete reference(boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException {
        Policy newPolicy = policy.reference(null, sharedPolicyFunctionEstimator, sharedMemory);
        ValueFunction newValueFunction = valueFunction.reference(newPolicy.getFunctionEstimator(), false, newPolicy.getFunctionEstimator().getMemory());
        return new SoftActorCriticDiscrete(getStateSynchronization(), getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), newPolicy.getFunctionEstimator(), (SoftQValueFunctionEstimator) newValueFunction, getParams());
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
    public SoftActorCriticDiscrete reference(boolean sharedPolicyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException {
        Policy newPolicy = policy.reference(null, sharedPolicyFunctionEstimator, sharedMemory);
        ValueFunction newValueFunction = valueFunction.reference(newPolicy.getFunctionEstimator(), sharedValueFunctionEstimator, newPolicy.getFunctionEstimator().getMemory());
        return new SoftActorCriticDiscrete(getStateSynchronization(), getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), newPolicy.getFunctionEstimator(), (SoftQValueFunctionEstimator) newValueFunction, getParams());
    }

}
