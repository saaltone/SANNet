/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.Environment;
import core.reinforcement.agent.StateSynchronization;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.ActionablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.value.QValueFunctionEstimator;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements Q Learning algorithm.<br>
 *
 */
public class DQNLearning extends AbstractQLearning {

    /**
     * Constructor for Q Learning.
     *
     * @param stateSynchronization reference to state synchronization.
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public DQNLearning(StateSynchronization stateSynchronization, Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator valueFunctionEstimator, String params) throws DynamicParamException, AgentException {
        super(stateSynchronization, environment, new ActionablePolicy(executablePolicyType, valueFunctionEstimator), new QValueFunctionEstimator(valueFunctionEstimator), params);
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
    public DQNLearning reference() throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException {
        return new DQNLearning(getStateSynchronization(), getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), valueFunction.reference().getFunctionEstimator(), getParams());
    }

    /**
     * Returns reference to algorithm.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public DQNLearning reference(boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException {
        return new DQNLearning(getStateSynchronization(), getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), valueFunction.reference(sharedValueFunctionEstimator, sharedMemory).getFunctionEstimator(), getParams());
    }

}
