/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.AgentException;
import core.reinforcement.Environment;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.ActionablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.value.ActionValueFunctionEstimator;
import utils.DynamicParamException;

/**
 * Class that defines Sarsa algorithm.
 *
 */
public class Sarsa extends AbstractQLearning {

    /**
     * Constructor for Sarsa.
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param valueFunctionEstimator reference to value function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if creation of executable policy fails.
     */
    public Sarsa(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator valueFunctionEstimator) throws DynamicParamException, AgentException {
        super(environment, new ActionablePolicy(executablePolicyType, valueFunctionEstimator), new ActionValueFunctionEstimator(valueFunctionEstimator));
    }

    /**
     * Constructor for Sarsa.
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if creation of executable policy fails.
     */
    public Sarsa(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator valueFunctionEstimator, String params) throws DynamicParamException, AgentException {
        super(environment, new ActionablePolicy(executablePolicyType, valueFunctionEstimator), new ActionValueFunctionEstimator(valueFunctionEstimator), params);
    }

}
