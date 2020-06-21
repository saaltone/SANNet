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

import java.util.TreeMap;

/**
 * Class that defines Q Learning algorithms.
 *
 */
public abstract class AbstractQLearning extends DeepAgent {

    /**
     * Constructor for Q Learning.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param buffer reference to buffer.
     * @param valueFunction reference to value function.
     */
    public AbstractQLearning(Environment environment, ActionablePolicy policy, Buffer buffer, ValueFunction valueFunction) {
        super(environment, policy, buffer, valueFunction);
    }

    /**
     * Constructor for Q Learning.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param buffer reference to buffer.
     * @param valueFunction reference to value function.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractQLearning(Environment environment, ActionablePolicy policy, Buffer buffer, ValueFunction valueFunction, String params) throws DynamicParamException {
        super(environment, policy, buffer, valueFunction, params);
    }

    /**
     * Updates value function of agent.
     *
     * @param samples samples used to update function estimator.
     * @param hasImportanceSamplingWeights if true samples contain importance sampling weights otherwise false.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void updateAgent(TreeMap<Integer, RLSample> samples, boolean hasImportanceSamplingWeights) throws MatrixException, NeuralNetworkException, DynamicParamException {
        valueFunction.updateFunctionEstimator(samples, hasImportanceSamplingWeights);
    }

}
