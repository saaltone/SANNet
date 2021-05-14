/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.Environment;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.executablepolicy.MCTSPolicy;
import core.reinforcement.policy.updateablepolicy.UpdateableMCTSPolicy;
import core.reinforcement.value.StateValueFunctionEstimator;
import utils.DynamicParamException;

/**
 * Class that defines MCTS learning algorithm.<br>
 *
 */
public class MCTSLearning extends AbstractPolicyGradient {

    /**
     * Constructor for MCTS learning
     *
     * @param environment reference to environment.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MCTSLearning(Environment environment, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator) throws DynamicParamException {
        super(environment, new UpdateableMCTSPolicy(policyFunctionEstimator), new StateValueFunctionEstimator(valueFunctionEstimator), "gamma = 1, updateValuePerEpisode = true");
    }

    /**
     * Constructor for MCTS learning
     *
     * @param environment reference to environment.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MCTSLearning(Environment environment, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator, String params) throws DynamicParamException {
        super(environment, new UpdateableMCTSPolicy(policyFunctionEstimator), new StateValueFunctionEstimator(valueFunctionEstimator), "gamma = 1, updateValuePerEpisode = true" + (params.isEmpty() ? "" : ", " + params));
    }

    /**
     * Constructor for MCTS learning
     *
     * @param environment reference to environment.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param mctsPolicy reference to MCTS policy.
     * @param valueFunctionEstimator reference to value function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MCTSLearning(Environment environment, FunctionEstimator policyFunctionEstimator, MCTSPolicy mctsPolicy, FunctionEstimator valueFunctionEstimator) throws DynamicParamException {
        super(environment, new UpdateableMCTSPolicy(policyFunctionEstimator, mctsPolicy), new StateValueFunctionEstimator(valueFunctionEstimator), "gamma = 1, updateValuePerEpisode = true");
    }

    /**
     * Constructor for MCTS learning
     *
     * @param environment reference to environment.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param mctsPolicy reference to MCTS policy.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MCTSLearning(Environment environment, FunctionEstimator policyFunctionEstimator, MCTSPolicy mctsPolicy, FunctionEstimator valueFunctionEstimator, String params) throws DynamicParamException {
        super(environment, new UpdateableMCTSPolicy(policyFunctionEstimator, mctsPolicy), new StateValueFunctionEstimator(valueFunctionEstimator), "gamma = 1, updateValuePerEpisode = true" + (params.isEmpty() ? "" : ", " + params));
    }

}
