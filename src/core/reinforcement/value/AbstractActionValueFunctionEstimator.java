/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.value;

import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.agent.State;
import utils.configurable.DynamicParamException;

/**
 * Implements abstract action value function estimator.<br>
 *
 */
public abstract class AbstractActionValueFunctionEstimator extends AbstractValueFunctionEstimator {

    /**
     * Constructor for abstract action value function estimator
     *
     * @param functionEstimator reference to function estimator.
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractActionValueFunctionEstimator(FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super(functionEstimator, params);
    }

    /**
     * Returns function index applying potential state action value offset.
     *
     * @param state state.
     * @return function index.
     */
    protected int getValueFunctionIndex(State state) {
        return state.action;
    }

}
