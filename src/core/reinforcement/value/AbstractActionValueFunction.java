/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.State;
import core.reinforcement.function.FunctionEstimator;
import utils.matrix.MatrixException;

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
     * Returns target value based on next state.
     *
     * @param nextState next state.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    public double getTargetValue(State nextState) throws NeuralNetworkException, MatrixException {
        int targetAction = getValueFunctionIndex(nextState);
        return targetAction == Integer.MIN_VALUE ? 0 : getFunctionEstimator().predictStateActionValues(nextState).getValue(getValueFunctionIndex(nextState), 0, 0);
    }

}
