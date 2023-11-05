/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.Environment;
import core.reinforcement.agent.StateSynchronization;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.ActionablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.value.ActionValueFunctionEstimator;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements Sarsa algorithm.<br>
 *
 */
public class Sarsa extends AbstractQLearning {

    /**
     * Constructor for Sarsa.
     *
     * @param stateSynchronization reference to state synchronization.
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public Sarsa(StateSynchronization stateSynchronization, Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator valueFunctionEstimator, String params) throws DynamicParamException, AgentException {
        super(stateSynchronization, environment, new ActionablePolicy(executablePolicyType, valueFunctionEstimator), new ActionValueFunctionEstimator(valueFunctionEstimator), params);
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
    public Sarsa reference() throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException {
        return new Sarsa(getStateSynchronization(), getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), valueFunction.reference().getFunctionEstimator(), getParams());
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
    public Sarsa reference(boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException {
        return new Sarsa(getStateSynchronization(), getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), valueFunction.reference(sharedValueFunctionEstimator, sharedMemory).getFunctionEstimator(), getParams());
    }

}
