/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.agent.Environment;
import core.reinforcement.agent.StateSynchronization;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.ActionablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.value.QTargetValueFunction;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements Double Q Learning (DDQN) algorithm.<br>
 *
 */
public class DDQNLearning extends AbstractQLearning {

    /**
     * Constructor for Double Q Learning (DDQN).
     *
     * @param stateSynchronization   reference to state synchronization.
     * @param environment            reference to environment.
     * @param executablePolicyType   executable policy type.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param memory                 reference to memory.
     * @param params                 parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public DDQNLearning(StateSynchronization stateSynchronization, Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator valueFunctionEstimator, Memory memory, String params) throws DynamicParamException, MatrixException {
        super(stateSynchronization, environment, new ActionablePolicy(executablePolicyType, valueFunctionEstimator, memory, params), new QTargetValueFunction(valueFunctionEstimator, params), memory, params);
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
    public DDQNLearning reference(boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException {
        Memory newMemory = sharedMemory ? memory : memory.reference();
        return new DDQNLearning(getStateSynchronization(), getEnvironment(), policy.getExecutablePolicy().getExecutablePolicyType(), valueFunction.reference(sharedValueFunctionEstimator).getFunctionEstimator(), newMemory, getParams());
    }

}
