/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.StateTransition;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements Q target value function estimator.<br>
 *
 */
public class QTargetValueFunctionEstimator extends AbstractActionValueFunctionEstimator {

    /**
     * Constructor for Q target value function estimator.
     *
     * @param functionEstimator reference to function estimator.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public QTargetValueFunctionEstimator(FunctionEstimator functionEstimator) throws IOException, ClassNotFoundException, DynamicParamException {
        super(functionEstimator);
        functionEstimator.setTargetFunctionEstimator();
    }

    /**
     * Constructor for Q target value function estimator.
     *
     * @param functionEstimator reference to value FunctionEstimator.
     * @param params parameters for Q target value function estimator.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
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
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public ValueFunction reference() throws DynamicParamException, IOException, ClassNotFoundException, MatrixException {
        return new QTargetValueFunctionEstimator(functionEstimator.reference(), getParams());
    }

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator, boolean sharedMemory) throws DynamicParamException, IOException, ClassNotFoundException, MatrixException {
        return new QTargetValueFunctionEstimator(sharedValueFunctionEstimator ? functionEstimator : functionEstimator.reference(sharedMemory), getParams());
    }

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param memory reference to memory.
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator, Memory memory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new QTargetValueFunctionEstimator(sharedValueFunctionEstimator ? functionEstimator : functionEstimator.reference(memory), getParams());
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
        return getValues(functionEstimator.getTargetFunctionEstimator(), nextStateTransition).getValue(functionEstimator.argmax(getValues(functionEstimator, nextStateTransition), nextStateTransition.environmentState.availableActions()), 0);
    }

}
