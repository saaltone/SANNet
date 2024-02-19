/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.agent.Environment;
import core.reinforcement.agent.StateSynchronization;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.updateablepolicy.UpdateableMCTSPolicy;
import core.reinforcement.value.PlainValueFunction;
import core.reinforcement.value.ValueFunction;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements MCTS Learning algorithm.<br>
 *
 */
public class MCTSLearning extends AbstractPolicyGradient {

    /**
     * Constructor for MCTS Learning
     *
     * @param stateSynchronization reference to state synchronization.
     * @param environment          reference to environment.
     * @param updateableMCTSPolicy reference to updateable MCTS policy.
     * @param plainValueFunction   reference to plain value function.
     * @param memory               reference to memory.
     * @param params               parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MCTSLearning(StateSynchronization stateSynchronization, Environment environment, UpdateableMCTSPolicy updateableMCTSPolicy, PlainValueFunction plainValueFunction, Memory memory, String params) throws DynamicParamException {
        super(stateSynchronization, environment, updateableMCTSPolicy, plainValueFunction, memory, params);
    }

    /**
     * Returns reference to algorithm.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedValueFunctionEstimator if true shared value function estimator is used otherwise new value function estimator is created.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public MCTSLearning reference(boolean sharedPolicyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException {
        Memory newMemory = getMemory(sharedMemory);
        ValueFunction newValueFunction = getValueFunction(sharedValueFunctionEstimator);
        Policy newPolicy = policy.reference(sharedPolicyFunctionEstimator, newMemory);
        return new MCTSLearning(getStateSynchronization(), getEnvironment(), (UpdateableMCTSPolicy) newPolicy, (PlainValueFunction) newValueFunction, newMemory, getParams());
    }

}
