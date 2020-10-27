/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.NeuralNetworkException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.StateTransition;
import utils.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashSet;

/**
 * Class that defines ActionValueFunctionEstimator (action value function with function estimator).
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
     * Returns action with potential state action value offset.
     *
     * @param action action.
     * @return updated action.
     */
    protected int getAction(int action) {
        return (isStateActionValueFunction() ? 1 : 0) + action;
    }

    /**
     * Returns target value based on next state.
     *
     * @param nextStateTransition next state transition.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return target value based on next state
     */
    public double getTargetValue(StateTransition nextStateTransition) throws NeuralNetworkException, MatrixException {
        return nextStateTransition.tdTarget;
    }

    /**
     * Returns max value of state given available actions.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return max value of state.
     */
    protected double max(Matrix stateValues, HashSet<Integer> availableActions) {
        return stateValues.getValue(getAction(argmax(stateValues, availableActions)), 0);
    }

    /**
     * Returns action with maximum state value given available actions.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return action with maximum state value.
     */
    protected int argmax(Matrix stateValues, HashSet<Integer> availableActions) {
        int maxAction = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int action : availableActions) {
            double actionValue = stateValues.getValue(getAction(action), 0);
            if (maxValue < actionValue) {
                maxValue = actionValue;
                maxAction = action;
            }
        }
        return maxAction;
    }

}
