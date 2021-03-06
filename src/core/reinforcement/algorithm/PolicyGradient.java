/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.*;
import core.reinforcement.policy.Policy;
import core.reinforcement.value.ValueFunction;
import utils.DynamicParamException;

/**
 * Class that defines generic policy gradient algorithm.<br>
 * Can be used to implement Actor Critic and REINFORCE algorithms.<br>
 *
 */
public class PolicyGradient extends AbstractPolicyGradient {

    /**
     * Constructor for PolicyGradient.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to valueFunction.
     */
    public PolicyGradient(Environment environment, Policy policy, ValueFunction valueFunction) {
        super(environment, policy, valueFunction);
    }

    /**
     * Constructor for PolicyGradient.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to valueFunction.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public PolicyGradient(Environment environment, Policy policy, ValueFunction valueFunction, String params) throws DynamicParamException {
        super(environment, policy, valueFunction, params);
    }

}
