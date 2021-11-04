/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.Environment;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.policy.updateablepolicy.UpdateableSoftQPolicy;
import core.reinforcement.value.SoftQValueFunctionEstimator;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Class that defines discrete Soft Actor Critic algorithm.<br>
 *
 */
public class SoftActorCriticDiscrete extends AbstractPolicyGradient {

    /**
     * Constructor for SoftActorCriticDiscrete
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param softQAlphaMatrix reference to soft Q alpha matrix.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public SoftActorCriticDiscrete(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator, Matrix softQAlphaMatrix) throws ClassNotFoundException, MatrixException, DynamicParamException, IOException, NeuralNetworkException, AgentException {
        super(environment, new UpdateableSoftQPolicy(executablePolicyType, policyFunctionEstimator, softQAlphaMatrix), new SoftQValueFunctionEstimator(policyFunctionEstimator, valueFunctionEstimator, softQAlphaMatrix));
    }

    /**
     * Constructor for SoftActorCriticDiscrete
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param softQAlphaMatrix reference to soft Q alpha matrix.
     * @param params parameters for agent.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public SoftActorCriticDiscrete(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator, Matrix softQAlphaMatrix, String params) throws ClassNotFoundException, MatrixException, DynamicParamException, IOException, NeuralNetworkException, AgentException {
        super(environment, new UpdateableSoftQPolicy(executablePolicyType, policyFunctionEstimator, softQAlphaMatrix), new SoftQValueFunctionEstimator(policyFunctionEstimator, valueFunctionEstimator, softQAlphaMatrix), params);
    }

    /**
     * Returns reference to algorithm.
     *
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws NeuralNetworkException throws exception if optimizer is of an unknown type.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public SoftActorCriticDiscrete reference() throws MatrixException, NeuralNetworkException, IOException, DynamicParamException, ClassNotFoundException, AgentException {
        return new SoftActorCriticDiscrete(getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), policy.reference().getFunctionEstimator(), valueFunction.reference().getFunctionEstimator(), new DMatrix(0), getParams());
    }

    /**
     * Returns reference to algorithm.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws NeuralNetworkException throws exception if optimizer is of an unknown type.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public SoftActorCriticDiscrete reference(boolean sharedPolicyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, NeuralNetworkException, IOException, DynamicParamException, ClassNotFoundException, AgentException {
        return new SoftActorCriticDiscrete(getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), policy.reference(sharedPolicyFunctionEstimator, sharedMemory).getFunctionEstimator(), valueFunction.reference(sharedValueFunctionEstimator, sharedMemory).getFunctionEstimator(), new DMatrix(0), getParams());
    }

}
