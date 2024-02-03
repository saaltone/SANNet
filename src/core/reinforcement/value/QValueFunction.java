/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.State;
import core.reinforcement.function.FunctionEstimator;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements Q value function (Q value function with function estimator).<br>
 *
 */
public class QValueFunction extends AbstractActionValueFunction {

    /**
     * Constructor for Q value function.
     *
     * @param functionEstimator reference to function estimator.
     * @param params parameters for value function.
     */
    public QValueFunction(FunctionEstimator functionEstimator, String params) {
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
        return new QValueFunction(sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(), getParams());
    }

    /**
     * Returns target value based on next state. Uses max value of next state.
     *
     * @param nextState next state.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public double getTargetValue(State nextState) throws NeuralNetworkException, MatrixException, DynamicParamException {
        return getStateValues(nextState).getValue(nextState.targetAction, 0, 0);
    }

    /**
     * Returns target action based on next state.
     *
     * @param nextState next state.
     * @return target action based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    protected int getTargetAction(State nextState) throws NeuralNetworkException, MatrixException {
        return getFunctionEstimator().argmax(getStateValues(nextState), nextState.environmentState.availableActions());
    }

}
