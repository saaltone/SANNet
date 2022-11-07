/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.Environment;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.policy.updateablepolicy.UpdateableBasicPolicy;
import core.reinforcement.value.StateValueFunctionEstimator;
import core.reinforcement.value.ValueFunction;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements actor critic algorithm.<br>
 *
 */
public class ActorCritic extends AbstractPolicyGradient {

    /**
     * Constructor for actor critic
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public ActorCritic(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator) throws DynamicParamException, AgentException {
        super(environment, new UpdateableBasicPolicy(executablePolicyType, policyFunctionEstimator), new StateValueFunctionEstimator(valueFunctionEstimator));
    }

    /**
     * Constructor for actor critic
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param params parameters
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public ActorCritic(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator, String params) throws DynamicParamException, AgentException {
        super(environment, new UpdateableBasicPolicy(executablePolicyType, policyFunctionEstimator), new StateValueFunctionEstimator(valueFunctionEstimator), params);
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
     * @throws NeuralNetworkException throws exception if starting of function estimator fails.
     */
    public ActorCritic reference() throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException, NeuralNetworkException {
        Policy newPolicy = policy.reference();
        ValueFunction newValueFunction = valueFunction.reference(false, policy.getFunctionEstimator().getMemory());
        return new ActorCritic(getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), newPolicy.getFunctionEstimator(), newValueFunction.getFunctionEstimator(), getParams());
    }

    /**
     * Returns reference to algorithm.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws NeuralNetworkException throws exception if starting of function estimator fails.
     */
    public ActorCritic reference(boolean sharedPolicyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException, NeuralNetworkException {
        Policy newPolicy = policy.reference(sharedPolicyFunctionEstimator, sharedMemory);
        ValueFunction newValueFunction = valueFunction.reference(false, policy.getFunctionEstimator().getMemory());
        return new ActorCritic(getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), newPolicy.getFunctionEstimator(), newValueFunction.getFunctionEstimator(), getParams());
    }

}
