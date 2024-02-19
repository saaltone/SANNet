/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.agent.Environment;
import core.reinforcement.agent.StateSynchronization;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.ActionablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.value.ActionValueFunction;
import core.reinforcement.value.ValueFunction;
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
     * @param stateSynchronization   reference to state synchronization.
     * @param environment            reference to environment.
     * @param executablePolicyType   executable policy type.
     * @param actionValueFunction    reference to action value function.
     * @param memory                 reference to memory.
     * @param params                 parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Sarsa(StateSynchronization stateSynchronization, Environment environment, ExecutablePolicyType executablePolicyType, ActionValueFunction actionValueFunction, Memory memory, String params) throws DynamicParamException, MatrixException {
        super(stateSynchronization, environment, new ActionablePolicy(executablePolicyType, actionValueFunction.getFunctionEstimator(), memory, params), actionValueFunction, memory, params);
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
     */
    public Sarsa reference(boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException {
        Memory newMemory = getMemory(sharedMemory);
        ValueFunction newValueFunction = getValueFunction(sharedValueFunctionEstimator);
        return new Sarsa(getStateSynchronization(), getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), (ActionValueFunction)newValueFunction, newMemory, getParams());
    }

}
