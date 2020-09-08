/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.NeuralNetworkException;
import core.reinforcement.*;
import core.reinforcement.policy.ActionablePolicy;
import core.reinforcement.value.ValueFunction;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

/**
 * Class that defines policy gradient algorithms.
 *
 */
public abstract class AbstractPolicyGradient extends DeepAgent {

    /**
     * Constructor for AbstractPolicyGradient.
     *
     * @param environment reference to environment.
     * @param actionablePolicy reference to policy.
     * @param valueFunction reference to value function.
     */
    public AbstractPolicyGradient(Environment environment, ActionablePolicy actionablePolicy, ValueFunction valueFunction) {
        super(environment, actionablePolicy, valueFunction);
    }

    /**
     * Constructor for AbstractPolicyGradient.
     *
     * @param environment reference to environment.
     * @param actionablePolicy reference to policy.
     * @param valueFunction reference to value function.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractPolicyGradient(Environment environment, ActionablePolicy actionablePolicy, ValueFunction valueFunction, String params) throws DynamicParamException {
        super(environment, actionablePolicy, valueFunction, params);
    }

    /**
     * Updates agent.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    protected void update() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException {
        valueFunction.update(this);
        policy.update(this);
        policy.update();
    }

}
