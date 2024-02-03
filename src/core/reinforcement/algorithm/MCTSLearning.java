/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.agent.Environment;
import core.reinforcement.agent.StateSynchronization;
import core.reinforcement.function.DirectFunctionEstimator;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.MCTSPolicy;
import core.reinforcement.policy.updateablepolicy.UpdateableMCTSPolicy;
import core.reinforcement.value.PlainValueFunction;
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
     * @param stateSynchronization    reference to state synchronization.
     * @param environment             reference to environment.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param memory                  reference to memory.
     * @param params                  parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MCTSLearning(StateSynchronization stateSynchronization, Environment environment, FunctionEstimator policyFunctionEstimator, Memory memory, String params) throws DynamicParamException, MatrixException {
        super(stateSynchronization, environment, new UpdateableMCTSPolicy(policyFunctionEstimator, new MCTSPolicy(), memory, params), new PlainValueFunction(new DirectFunctionEstimator(policyFunctionEstimator, params), params), memory, "gamma = 1" + (params.isEmpty() ? "" : ", " + params));
    }

    /**
     * Constructor for MCTS Learning
     *
     * @param stateSynchronization    reference to state synchronization.
     * @param environment             reference to environment.
     * @param mctsPolicy              reference to MCTS policy.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param memory                  reference to memory.
     * @param params                  parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MCTSLearning(StateSynchronization stateSynchronization, Environment environment, MCTSPolicy mctsPolicy, FunctionEstimator policyFunctionEstimator, Memory memory, String params) throws DynamicParamException, MatrixException {
        super(stateSynchronization, environment, new UpdateableMCTSPolicy(policyFunctionEstimator, mctsPolicy, memory, null), new PlainValueFunction(new DirectFunctionEstimator(policyFunctionEstimator, params), params), memory, "gamma = 1" + (params.isEmpty() ? "" : ", " + params));
    }

    /**
     * Returns reference to algorithm.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedValueFunctionEstimator if true shared value function estimator is used otherwise new policy function estimator is created.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public MCTSLearning reference(boolean sharedPolicyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException {
        Memory newMemory = sharedMemory ? memory : memory.reference();
        Policy newPolicy = policy.reference(sharedPolicyFunctionEstimator, newMemory);
        return new MCTSLearning(getStateSynchronization(), getEnvironment(), (MCTSPolicy)newPolicy.getExecutablePolicy(), newPolicy.getFunctionEstimator(), newMemory, getParams());
    }

}
