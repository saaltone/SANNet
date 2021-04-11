/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.NeuralNetworkException;
import core.reinforcement.*;
import core.reinforcement.policy.Policy;
import core.reinforcement.value.ValueFunction;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

/**
 * Class that defines Q Learning algorithms.<br>
 *
 */
public abstract class AbstractQLearning extends DeepAgent {

    /**
     * Constructor for AbstractQLearning.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to value function.
     */
    public AbstractQLearning(Environment environment, Policy policy, ValueFunction valueFunction) {
        initialize(environment, policy, valueFunction);
    }

    /**
     * Constructor for AbstractQLearning.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to value function.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractQLearning(Environment environment, Policy policy, ValueFunction valueFunction, String params) throws DynamicParamException {
        initialize(environment, policy, valueFunction, params);
    }

    /**
     * Updates value function of agent.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if memory instances of value and policy function are not equal.
     */
    protected void updateFunctionEstimator() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException {
        if(valueFunction.readyToUpdate(this)) {
            valueFunction.sample();
            if (!updateValuePerEpisode) valueFunction.update();
            valueFunction.updateFunctionEstimator();
            valueFunction.resetFunctionEstimator();
        }
    }

}
