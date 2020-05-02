/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement.algorithm;

import core.reinforcement.*;
import core.reinforcement.policy.UpdateablePolicy;
import core.reinforcement.value.ValueFunction;
import utils.DynamicParamException;

/**
 * Class that defines generic PolicyGradient algorithm.<br>
 * Can be used to implement Actor Critic and baselined REINFORCE algorithms without bootstrapping with ValueFunctionEstimator or non-baselined REINFORCE with PlainValueFunction.<br>
 *
 */
public class PolicyGradient extends AbstractPolicyGradient {

    /**
     * Constructor for PolicyGradient.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param buffer reference to buffer.
     * @param valueFunction reference to valueFunction.
     */
    public PolicyGradient(Environment environment, UpdateablePolicy policy, Buffer buffer, ValueFunction valueFunction) {
        super(environment, policy, buffer, valueFunction);
    }

    /**
     * Constructor for PolicyGradient.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param buffer reference to buffer.
     * @param valueFunction reference to valueFunction.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public PolicyGradient(Environment environment, UpdateablePolicy policy, Buffer buffer, ValueFunction valueFunction, String params) throws DynamicParamException {
        super(environment, policy, buffer, valueFunction, params);
    }

}
