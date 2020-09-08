/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.Environment;
import core.reinforcement.policy.ActionablePolicy;
import core.reinforcement.value.StateValueFunctionEstimator;
import utils.DynamicParamException;

/**
 * Class that defines actor critic algorithm.
 *
 */
public class ActorCritic extends AbstractPolicyGradient {

    /**
     * Constructor for ActorCritic
     *
     * @param environment reference to environment.
     * @param actionablePolicy reference to policy.
     * @param stateValueFunctionEstimator reference to state value function estimator.
     */
    public ActorCritic(Environment environment, ActionablePolicy actionablePolicy, StateValueFunctionEstimator stateValueFunctionEstimator) {
        super(environment, actionablePolicy, stateValueFunctionEstimator);
    }

    /**
     * Constructor for ActorCritic
     *
     * @param environment reference to environment.
     * @param actionablePolicy reference to policy.
     * @param stateValueFunctionEstimator reference to state value function estimator.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ActorCritic(Environment environment, ActionablePolicy actionablePolicy, StateValueFunctionEstimator stateValueFunctionEstimator, String params) throws DynamicParamException {
        super(environment, actionablePolicy, stateValueFunctionEstimator, params);
    }

}
