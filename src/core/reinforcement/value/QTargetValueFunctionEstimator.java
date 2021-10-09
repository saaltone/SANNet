/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.StateTransition;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Class that defines QTargetValueFunctionEstimator (Q value function with target function estimator).<br>
 *
 */
public class QTargetValueFunctionEstimator extends AbstractActionValueFunctionEstimator {

    /**
     * Constructor for QTargetValueFunctionEstimator.
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public QTargetValueFunctionEstimator(FunctionEstimator functionEstimator) throws IOException, ClassNotFoundException, DynamicParamException {
        super(functionEstimator);
        functionEstimator.setTargetFunctionEstimator();
    }

    /**
     * Constructor for QTargetValueFunctionEstimator.
     *
     * @param functionEstimator reference to value FunctionEstimator.
     * @param params parameters for QTargetValueFunctionEstimator.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public QTargetValueFunctionEstimator(FunctionEstimator functionEstimator, String params) throws IOException, ClassNotFoundException, DynamicParamException {
        super(functionEstimator, params);
        functionEstimator.setTargetFunctionEstimator();
    }

    /**
     * Returns reference to value function.
     *
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ValueFunction reference() throws DynamicParamException, IOException, ClassNotFoundException {
        return new QTargetValueFunctionEstimator(functionEstimator, getParams());
    }

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if optimizer is of an unknown type.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator, boolean sharedMemory) throws DynamicParamException, IOException, ClassNotFoundException, NeuralNetworkException {
        return new QTargetValueFunctionEstimator(sharedValueFunctionEstimator ? functionEstimator : functionEstimator.reference(sharedMemory), getParams());
    }

    /**
     * Returns target value based on next state. Uses target action with maximal value as defined by target FunctionEstimator.
     *
     * @param nextStateTransition next state transition.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getTargetValue(StateTransition nextStateTransition) throws NeuralNetworkException, MatrixException {
        return getValues(functionEstimator.getTargetFunctionEstimator(), nextStateTransition.environmentState.state()).getValue(functionEstimator.argmax(getValues(functionEstimator, nextStateTransition.environmentState.state()), nextStateTransition.environmentState.availableActions()), 0);
    }

}
