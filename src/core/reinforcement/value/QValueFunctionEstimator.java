/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.memory.Memory;
import core.reinforcement.agent.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements Q value function estimator (Q value function with function estimator).<br>
 *
 */
public class QValueFunctionEstimator extends AbstractActionValueFunctionEstimator {

    /**
     * Constructor for Q value function estimator.
     *
     * @param functionEstimator reference to function estimator.
     */
    public QValueFunctionEstimator(FunctionEstimator functionEstimator) {
        super(functionEstimator);
    }

    /**
     * Constructor for Q value function estimator.
     *
     * @param functionEstimator reference to function estimator.
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
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public ValueFunction reference() throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new QValueFunctionEstimator(getFunctionEstimator().reference(), getParams());
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
    public ValueFunction reference(boolean sharedValueFunctionEstimator, boolean sharedMemory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new QValueFunctionEstimator(sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(sharedMemory), getParams());
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
        return new QValueFunctionEstimator(sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(memory), getParams());
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
        return getFunctionEstimator().max(getValues(getFunctionEstimator(), nextStateTransition), nextStateTransition.environmentState.availableActions());
    }

}
