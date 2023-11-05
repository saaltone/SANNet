/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.Environment;
import core.reinforcement.agent.StateSynchronization;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.MCTSPolicy;
import core.reinforcement.policy.updateablepolicy.UpdateableMCTSPolicy;
import core.reinforcement.value.StateValueFunctionEstimator;
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
     * @param environment reference to environment.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public MCTSLearning(StateSynchronization stateSynchronization, Environment environment, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator, String params) throws DynamicParamException, AgentException {
        super(stateSynchronization, environment, new UpdateableMCTSPolicy(policyFunctionEstimator), new StateValueFunctionEstimator(valueFunctionEstimator), "gamma = 1, updateValuePerEpisode = true" + (params.isEmpty() ? "" : ", " + params));
    }

    /**
     * Constructor for MCTS Learning
     *
     * @param stateSynchronization reference to state synchronization.
     * @param environment reference to environment.
     * @param mctsPolicy reference to MCTS policy.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public MCTSLearning(StateSynchronization stateSynchronization, Environment environment, MCTSPolicy mctsPolicy, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator, String params) throws DynamicParamException, AgentException {
        super(stateSynchronization, environment, new UpdateableMCTSPolicy(policyFunctionEstimator, mctsPolicy, null), new StateValueFunctionEstimator(valueFunctionEstimator), "gamma = 1, updateValuePerEpisode = true" + (params.isEmpty() ? "" : ", " + params));
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
    public MCTSLearning reference() throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException {
        Policy newPolicy = policy.reference(null);
        ValueFunction newValueFunction = valueFunction.reference(false, newPolicy.getFunctionEstimator().getMemory());
        return new MCTSLearning(getStateSynchronization(), getEnvironment(), (MCTSPolicy)newPolicy.getExecutablePolicy(), newPolicy.getFunctionEstimator(), newValueFunction.getFunctionEstimator(), getParams());
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
     */
    public MCTSLearning reference(boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException {
        Policy newPolicy = policy.reference(null, sharedPolicyFunctionEstimator, sharedMemory);
        ValueFunction newValueFunction = valueFunction.reference(false, newPolicy.getFunctionEstimator().getMemory());
        return new MCTSLearning(getStateSynchronization(), getEnvironment(), (MCTSPolicy)newPolicy.getExecutablePolicy(), newPolicy.getFunctionEstimator(), newValueFunction.getFunctionEstimator(), getParams());
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
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public MCTSLearning reference(boolean sharedPolicyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException {
        Policy newPolicy = policy.reference(null, sharedPolicyFunctionEstimator, sharedMemory);
        ValueFunction newValueFunction = valueFunction.reference(sharedValueFunctionEstimator, newPolicy.getFunctionEstimator().getMemory());
        return new MCTSLearning(getStateSynchronization(), getEnvironment(), (MCTSPolicy)newPolicy.getExecutablePolicy(), newPolicy.getFunctionEstimator(), newValueFunction.getFunctionEstimator(), getParams());
    }

}
