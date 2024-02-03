/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.Environment;
import core.reinforcement.agent.StateSynchronization;
import core.reinforcement.function.DirectFunctionEstimator;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.policy.updateablepolicy.UpdateableBasicPolicy;
import core.reinforcement.value.PlainValueFunction;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements REINFORCE algorithm.<br>
 *
 */
public class REINFORCE extends AbstractPolicyGradient {

    /**
     * Constructor for REINFORCE
     *
     * @param stateSynchronization reference to state synchronization.
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public REINFORCE(StateSynchronization stateSynchronization, Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, String params) throws DynamicParamException, AgentException {
        super(stateSynchronization, environment, new UpdateableBasicPolicy(executablePolicyType, policyFunctionEstimator), new PlainValueFunction(new DirectFunctionEstimator(policyFunctionEstimator)), params);
    }

    /**
     * Returns reference to algorithm.
     *
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if soft Q alpha matrix is non-scalar matrix.
     */
    public REINFORCE reference() throws IOException, DynamicParamException, ClassNotFoundException, AgentException, MatrixException {
        return new REINFORCE(getStateSynchronization(), getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), policy.reference(null).getFunctionEstimator(), getParams());
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
     * @throws AgentException throws exception if soft Q alpha matrix is non-scalar matrix.
     */
    public REINFORCE reference(boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws IOException, DynamicParamException, ClassNotFoundException, AgentException, MatrixException {
        return new REINFORCE(getStateSynchronization(), getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), policy.reference(null, sharedPolicyFunctionEstimator, sharedMemory).getFunctionEstimator(), getParams());
    }

    /**
     * Returns reference to abstract policy gradient algorithm.
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
    public AbstractPolicyGradient reference(boolean sharedPolicyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException {
        return new REINFORCE(getStateSynchronization(), getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), policy.reference(null, sharedPolicyFunctionEstimator, sharedMemory).getFunctionEstimator(), getParams());
    }

}
