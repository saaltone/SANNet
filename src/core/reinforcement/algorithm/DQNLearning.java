/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement.algorithm;

import core.reinforcement.Buffer;
import core.reinforcement.Environment;
import core.reinforcement.policy.ActionablePolicy;
import core.reinforcement.value.QValueFunctionEstimator;
import utils.DynamicParamException;

/**
 * Class that defines Q Learning with Function Estimator.
 *
 */
public class DQNLearning extends AbstractQLearning {

    /**
     * Constructor for DQNLearning.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param buffer reference to buffer.
     * @param qValueFunctionEstimator reference to qValueFunctionEstimator.
     */
    public DQNLearning(Environment environment, ActionablePolicy policy, Buffer buffer, QValueFunctionEstimator qValueFunctionEstimator) {
        super(environment, policy, buffer, qValueFunctionEstimator);
    }

    /**
     * Constructor for DQNLearning.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param buffer reference to buffer.
     * @param qValueFunctionEstimator reference to qValueFunctionEstimator.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DQNLearning(Environment environment, ActionablePolicy policy, Buffer buffer, QValueFunctionEstimator qValueFunctionEstimator, String params) throws DynamicParamException {
        super(environment, policy, buffer, qValueFunctionEstimator, params);
    }

}
