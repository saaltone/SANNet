/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.Environment;
import core.reinforcement.policy.Policy;
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
     * @param policy reference to policy.
     * @param stateValueFunctionEstimator reference to state value function estimator.
     */
    public ActorCritic(Environment environment, Policy policy, StateValueFunctionEstimator stateValueFunctionEstimator) {
        super(environment, policy, stateValueFunctionEstimator);
    }

    /**
     * Constructor for ActorCritic
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param stateValueFunctionEstimator reference to state value function estimator.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ActorCritic(Environment environment, Policy policy, StateValueFunctionEstimator stateValueFunctionEstimator, String params) throws DynamicParamException {
        super(environment, policy, stateValueFunctionEstimator, params);
    }

}
