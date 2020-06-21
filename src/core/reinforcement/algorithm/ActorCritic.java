/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.Buffer;
import core.reinforcement.Environment;
import core.reinforcement.policy.UpdateablePolicy;
import core.reinforcement.value.ValueFunctionEstimator;
import utils.DynamicParamException;

/**
 * Class that defines actor critic algorithm with Function Estimator.
 *
 */
public class ActorCritic extends AbstractPolicyGradient {

    /**
     * Constructor for Actor Critic
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param buffer reference to buffer.
     * @param valueFunctionEstimator reference to value function.
     */
    public ActorCritic(Environment environment, UpdateablePolicy policy, Buffer buffer, ValueFunctionEstimator valueFunctionEstimator) {
        super(environment, policy, buffer, valueFunctionEstimator);
    }

    /**
     * Constructor for Actor Critic
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param buffer reference to buffer.
     * @param valueFunctionEstimator reference to value function.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ActorCritic(Environment environment, UpdateablePolicy policy, Buffer buffer, ValueFunctionEstimator valueFunctionEstimator, String params) throws DynamicParamException {
        super(environment, policy, buffer, valueFunctionEstimator, params);
    }

}
