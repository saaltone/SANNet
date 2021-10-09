/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

/**
 * Class that defines QValueFunctionEstimator (Q value function with function estimator).<br>
 *
 */
public class QValueFunctionEstimator extends AbstractActionValueFunctionEstimator {

    /**
     * Constructor for QValueFunctionEstimator.
     *
     * @param functionEstimator reference to FunctionEstimator.
     */
    public QValueFunctionEstimator(FunctionEstimator functionEstimator) {
        super(functionEstimator);
    }

    /**
     * Constructor for QValueFunctionEstimator.
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public QValueFunctionEstimator(FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super(functionEstimator, params);
    }

    /**
     * Returns reference to value function.
     *
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ValueFunction reference() throws DynamicParamException {
        return new QValueFunctionEstimator(functionEstimator, getParams());
    }

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if optimizer is of an unknown type.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator, boolean sharedMemory) throws DynamicParamException, NeuralNetworkException {
        return new QValueFunctionEstimator(sharedValueFunctionEstimator ? functionEstimator : functionEstimator.reference(sharedMemory), getParams());
    }

    /**
     * Returns target value based on next state. Uses max value of next state.
     *
     * @param nextStateTransition next state transition.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getTargetValue(StateTransition nextStateTransition) throws NeuralNetworkException, MatrixException {
        return functionEstimator.max(getValues(functionEstimator, nextStateTransition.environmentState.state()), nextStateTransition.environmentState.availableActions());
    }

}
