/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import utils.DynamicParamException;

/**
 * Class that defines StateValueFunctionEstimator (state value function with function estimator).<br>
 *
 */
public class StateValueFunctionEstimator extends AbstractValueFunctionEstimator {

    /**
     * Constructor for StateValueFunctionEstimator
     *
     * @param functionEstimator reference to FunctionEstimator.
     */
    public StateValueFunctionEstimator(FunctionEstimator functionEstimator) {
        super(1, functionEstimator);
    }

    /**
     * Constructor for StateValueFunctionEstimator
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public StateValueFunctionEstimator(FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super(1, functionEstimator, params);
    }

    /**
     * Returns action. Always 0.
     *
     * @param action action.
     * @return updated action.
     */
    protected int getAction(int action) {
        return 0;
    }

    /**
     * Returns target value based on next state.
     *
     * @param nextStateTransition next state transition.
     * @return target value based on next state
     */
    public double getTargetValue(StateTransition nextStateTransition) {
        return nextStateTransition.tdTarget;
    }

}
