/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.Environment;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.policy.updateablepolicy.UpdateableSoftQPolicy;
import core.reinforcement.value.SoftQValueFunctionEstimator;
import core.reinforcement.value.ValueFunction;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
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
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param softQAlphaMatrix reference to soft Q alpha matrix.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws NeuralNetworkException throws exception if starting of function estimator fails.
     */
    public SoftActorCriticDiscrete(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator, Matrix softQAlphaMatrix) throws ClassNotFoundException, MatrixException, DynamicParamException, IOException, AgentException, NeuralNetworkException {
        super(environment, new UpdateableSoftQPolicy(executablePolicyType, policyFunctionEstimator, softQAlphaMatrix), new SoftQValueFunctionEstimator(policyFunctionEstimator, valueFunctionEstimator, softQAlphaMatrix));
    }

    /**
     * Constructor for discrete Soft Actor Critic
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param softQAlphaMatrix reference to soft Q alpha matrix.
     * @param params parameters for agent.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws NeuralNetworkException throws exception if starting of function estimator fails.
     */
    public SoftActorCriticDiscrete(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator, Matrix softQAlphaMatrix, String params) throws ClassNotFoundException, MatrixException, DynamicParamException, IOException, AgentException, NeuralNetworkException {
        super(environment, new UpdateableSoftQPolicy(executablePolicyType, policyFunctionEstimator, softQAlphaMatrix), new SoftQValueFunctionEstimator(policyFunctionEstimator, valueFunctionEstimator, softQAlphaMatrix), params);
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
    public SoftActorCriticDiscrete reference() throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException, NeuralNetworkException {
        Policy newPolicy = policy.reference();
        ValueFunction newValueFunction = valueFunction.reference(false, policy.getFunctionEstimator().getMemory());
        return new SoftActorCriticDiscrete(getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), newPolicy.getFunctionEstimator(), newValueFunction.getFunctionEstimator(), new DMatrix(0), getParams());
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
     * @throws NeuralNetworkException throws exception if starting of function estimator fails.
     */
    public SoftActorCriticDiscrete reference(boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException, NeuralNetworkException {
        Policy newPolicy = policy.reference(sharedPolicyFunctionEstimator, sharedMemory);
        ValueFunction newValueFunction = valueFunction.reference(false, policy.getFunctionEstimator().getMemory());
        return new SoftActorCriticDiscrete(getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), newPolicy.getFunctionEstimator(), newValueFunction.getFunctionEstimator(), new DMatrix(0), getParams());
    }

}
