/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.value;

import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.StateTransition;
import utils.DynamicParamException;

/**
 * Class that defines ActionValueFunctionEstimator (action value function with function estimator).<br>
 *
 */
public class ActionValueFunctionEstimator extends AbstractValueFunctionEstimator {

    /**
     * Constructor for ActionValueFunctionEstimator
     *
     * @param functionEstimator reference to FunctionEstimator.
     */
    public ActionValueFunctionEstimator(FunctionEstimator functionEstimator) {
        super(functionEstimator.getNumberOfActions(), functionEstimator);
    }

    /**
     * Constructor for ActionValueFunctionEstimator
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ActionValueFunctionEstimator(FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super(functionEstimator.getNumberOfActions(), functionEstimator, params);
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
