/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.*;
import core.reinforcement.policy.ActionablePolicy;
import core.reinforcement.value.ValueFunction;
import utils.DynamicParamException;

/**
 * Class that defines generic PolicyGradient algorithm.<br>
 * Can be used to implement Actor Critic and REINFORCE algorithms.<br>
 *
 */
public class PolicyGradient extends AbstractPolicyGradient {

    /**
     * Constructor for PolicyGradient.
     *
     * @param environment reference to environment.
     * @param actionablePolicy reference to policy.
     * @param valueFunction reference to valueFunction.
     */
    public PolicyGradient(Environment environment, ActionablePolicy actionablePolicy, ValueFunction valueFunction) {
        super(environment, actionablePolicy, valueFunction);
    }

    /**
     * Constructor for PolicyGradient.
     *
     * @param environment reference to environment.
     * @param actionablePolicy reference to policy.
     * @param valueFunction reference to valueFunction.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public PolicyGradient(Environment environment, ActionablePolicy actionablePolicy, ValueFunction valueFunction, String params) throws DynamicParamException {
        super(environment, actionablePolicy, valueFunction, params);
    }

}
