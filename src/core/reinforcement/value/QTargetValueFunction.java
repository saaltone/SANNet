/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.agent.State;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements target Q value function.<br>
 *
 */
public class QTargetValueFunction extends QValueFunction {

    /**
     * Constructor for target Q value function.
     *
     * @param functionEstimator reference to value FunctionEstimator.
     * @param params parameters for target Q value function estimator.
     */
    public QTargetValueFunction(FunctionEstimator functionEstimator, String params) {
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
        return new QTargetValueFunction(sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(), getParams());
    }

    /**
     * Returns target value based on next state. Uses target action with maximal value as defined by target FunctionEstimator.
     *
     * @param nextState next state.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public double getTargetValue(State nextState) throws NeuralNetworkException, MatrixException, DynamicParamException {
        return getFunctionEstimator().predictTargetStateActionValues(nextState).getValue(nextState.targetAction, 0, 0);
    }

}
