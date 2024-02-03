/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.value;

import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.agent.State;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements action value function.<br>
 *
 */
public class ActionValueFunction extends AbstractActionValueFunction {

    /**
     * Constructor for action value function
     *
     * @param functionEstimator reference to function estimator.
     * @param params parameters for value function.
     */
    public ActionValueFunction(FunctionEstimator functionEstimator, String params) {
        super(functionEstimator, params);
    }

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @return reference to value function.
     * @throws IOException            throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws MatrixException        throws exception if neural network has less output than actions.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new ActionValueFunction(sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(), getParams());
    }

    /**
     * Returns target value based on next state.
     *
     * @param nextState next state.
     * @return target value based on next state
     */
    public double getTargetValue(State nextState) {
        return nextState.value;
    }

    /**
     * Returns target action based on next state.
     *
     * @param nextState next state.
     * @return target action based on next state
     */
    protected int getTargetAction(State nextState) {
        return nextState.action;
    }

}
