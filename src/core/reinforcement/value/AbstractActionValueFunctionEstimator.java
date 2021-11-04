/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.value;

import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.StateTransition;
import utils.configurable.DynamicParamException;

/**
 * Class that defines AbstractActionValueFunctionEstimator (action value function with function estimator).<br>
 *
 */
public abstract class AbstractActionValueFunctionEstimator extends AbstractValueFunctionEstimator {

    /**
     * Constructor for AbstractActionValueFunctionEstimator
     *
     * @param functionEstimator reference to FunctionEstimator.
     */
    public AbstractActionValueFunctionEstimator(FunctionEstimator functionEstimator) {
        super(functionEstimator.getNumberOfActions(), functionEstimator);
    }

    /**
     * Constructor for AbstractActionValueFunctionEstimator
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractActionValueFunctionEstimator(FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super(functionEstimator.getNumberOfActions(), functionEstimator, params);
    }

    /**
     * Returns function index applying potential state action value offset.
     *
     * @param stateTransition state transition.
     * @return function index.
     */
    protected int getFunctionIndex(StateTransition stateTransition) {
        return stateTransition.action;
    }

}
