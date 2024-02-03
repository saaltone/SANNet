/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.value;

import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.agent.State;

/**
 * Implements abstract action value function.<br>
 *
 */
public abstract class AbstractActionValueFunction extends AbstractValueFunctionEstimator {

    /**
     * Constructor for abstract action value function
     *
     * @param functionEstimator reference to function.
     * @param params parameters for value function.
     */
    public AbstractActionValueFunction(FunctionEstimator functionEstimator, String params) {
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
