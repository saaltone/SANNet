/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.agent.State;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements target Q value function estimator.<br>
 *
 */
public class QTargetValueFunctionEstimator extends AbstractActionValueFunctionEstimator {

    /**
     * Constructor for target Q value function estimator.
     *
     * @param functionEstimator reference to function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public QTargetValueFunctionEstimator(FunctionEstimator functionEstimator) throws DynamicParamException {
        this(functionEstimator, null);
    }

    /**
     * Constructor for target Q value function estimator.
     *
     * @param functionEstimator reference to value FunctionEstimator.
     * @param params parameters for target Q value function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public QTargetValueFunctionEstimator(FunctionEstimator functionEstimator, String params) throws DynamicParamException {
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
    public ValueFunction reference() throws DynamicParamException, IOException, ClassNotFoundException, MatrixException {
        return new QTargetValueFunctionEstimator(getFunctionEstimator().reference(), getParams());
    }

    /**
     * Returns reference to value function.
     *
     * @param policyFunctionEstimator reference to policy function estimator.
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public ValueFunction reference(FunctionEstimator policyFunctionEstimator) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new QTargetValueFunctionEstimator(getFunctionEstimator().reference(), getParams());
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
        return new QTargetValueFunctionEstimator(sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(sharedMemory), getParams());
    }

    /**
     * Returns reference to value function.
     *
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public ValueFunction reference(FunctionEstimator policyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new QTargetValueFunctionEstimator(sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(sharedMemory), getParams());
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
        return new QTargetValueFunctionEstimator(sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(memory), getParams());
    }

    /**
     * Returns reference to value function.
     *
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param memory reference to memory.
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public ValueFunction reference(FunctionEstimator policyFunctionEstimator, boolean sharedValueFunctionEstimator, Memory memory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new QTargetValueFunctionEstimator(sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(memory), getParams());
    }

    /**
     * Returns target value based on next state. Uses target action with maximal value as defined by target FunctionEstimator.
     *
     * @param nextState next state.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getTargetValue(State nextState) throws NeuralNetworkException, MatrixException {
        return getTargetValue(getFunctionEstimator(), nextState);
    }

    /**
     * Returns target value based on next state. Uses target action with maximal value as defined by target FunctionEstimator.
     *
     * @param currentFunctionEstimator current function estimator.
     * @param nextState next state.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected double getTargetValue(FunctionEstimator currentFunctionEstimator, State nextState) throws NeuralNetworkException, MatrixException {
        return getTargetValues(currentFunctionEstimator, nextState).getValue(currentFunctionEstimator.argmax(currentFunctionEstimator.predictStateActionValues(nextState), nextState.environmentState.availableActions()), 0, 0);
    }

    /**
     * Returns values for state using target value estimator.
     *
     * @param currentFunctionEstimator current function estimator.
     * @param state state.
     * @return values for state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix getTargetValues(FunctionEstimator currentFunctionEstimator, State state) throws MatrixException, NeuralNetworkException {
        return currentFunctionEstimator.predictTargetStateActionValues(state);
    }

}
