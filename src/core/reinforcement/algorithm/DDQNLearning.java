/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.Environment;
import core.reinforcement.policy.ActionablePolicy;
import core.reinforcement.value.QTargetValueFunctionEstimator;
import utils.DynamicParamException;

/**
 * Class that defines Double Q Learning.
 *
 */
public class DDQNLearning extends AbstractQLearning {

    /**
     * Constructor for DDQNLearning.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param qTargetValueFunctionEstimator Q target value function estimator
     */
    public DDQNLearning(Environment environment, ActionablePolicy policy, QTargetValueFunctionEstimator qTargetValueFunctionEstimator) {
        super(environment, policy, qTargetValueFunctionEstimator);
    }

    /**
     * Constructor for DDQNLearning.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param qTargetValueFunctionEstimator Q target value function estimator
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DDQNLearning(Environment environment, ActionablePolicy policy, QTargetValueFunctionEstimator qTargetValueFunctionEstimator, String params) throws DynamicParamException {
        super(environment, policy, qTargetValueFunctionEstimator, params);
    }

}
