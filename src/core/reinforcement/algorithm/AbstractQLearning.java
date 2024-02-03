/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.*;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.ActionablePolicy;
import core.reinforcement.value.ValueFunction;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.TreeSet;

/**
 * Implements abstract Q Learning functionality.<br>
 *
 */
public abstract class AbstractQLearning extends DeepAgent {

    /**
     * Constructor for abstract Q Learning.
     *
     * @param stateSynchronization reference to state synchronization.
     * @param environment          reference to environment.
     * @param actionablePolicy     reference to actionablePolicy.
     * @param valueFunction        reference to value function.
     * @param memory               reference to memory.
     * @param params               parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractQLearning(StateSynchronization stateSynchronization, Environment environment, ActionablePolicy actionablePolicy, ValueFunction valueFunction, Memory memory, String params) throws DynamicParamException {
        super(stateSynchronization, environment, actionablePolicy, valueFunction, memory, params);
    }

    /**
     * Updates value function of agent.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    protected void updateFunctionEstimator() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException {
        TreeSet<State> sampledStates = memory.sample();
        if (sampledStates != null && !sampledStates.isEmpty()) {
            if(valueFunction.readyToUpdate(this)) {
                valueFunction.update(sampledStates);
                valueFunction.updateFunctionEstimator(sampledStates);
            }

            if (memory.readyToUpdate(this)) {
                memory.update();
                memory.reset();
            }
        }

    }

    /**
     * Returns reference to abstract Q Learning algorithm.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public abstract AbstractQLearning reference(boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException;

}
