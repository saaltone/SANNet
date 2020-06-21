/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.Buffer;
import core.reinforcement.Environment;
import core.reinforcement.policy.ActionablePolicy;
import core.reinforcement.value.ValueFunctionEstimator;
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
     * @param buffer reference to buffer.
     * @param valueFunctionEstimator reference to valueFunctionEstimator.
     */
    public Sarsa(Environment environment, ActionablePolicy policy, Buffer buffer, ValueFunctionEstimator valueFunctionEstimator) {
        super(environment, policy, buffer, valueFunctionEstimator);
    }

    /**
     * Constructor for Sarsa.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param buffer reference to buffer.
     * @param valueFunctionEstimator reference to valueFunctionEstimator.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Sarsa(Environment environment, ActionablePolicy policy, Buffer buffer, ValueFunctionEstimator valueFunctionEstimator, String params) throws DynamicParamException {
        super(environment, policy, buffer, valueFunctionEstimator, params);
    }
}
