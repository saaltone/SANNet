/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.Environment;
import core.reinforcement.policy.ActionablePolicy;
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
     * @param policy reference to policy.
     * @param actionValueFunctionEstimator reference to valueFunctionEstimator.
     */
    public Sarsa(Environment environment, ActionablePolicy policy, ActionValueFunctionEstimator actionValueFunctionEstimator) {
        super(environment, policy, actionValueFunctionEstimator);
    }

    /**
     * Constructor for Sarsa.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param actionValueFunctionEstimator reference to valueFunctionEstimator.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Sarsa(Environment environment, ActionablePolicy policy, ActionValueFunctionEstimator actionValueFunctionEstimator, String params) throws DynamicParamException {
        super(environment, policy, actionValueFunctionEstimator, params);
    }
}
