/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.function;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.AgentException;
import core.reinforcement.memory.Memory;
import core.reinforcement.agent.StateTransition;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements direct function estimator (proxy for memory) to be used with plain value function.<br>
 *
 */
public class DirectFunctionEstimator extends AbstractFunctionEstimator {

    /**
     * Constructor for direct function estimator.
     *
     * @param memory memory reference.
     * @param numberOfStates number of states for direct function estimator
     * @param numberOfActions number of actions for direct function estimator
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DirectFunctionEstimator(Memory memory, int numberOfStates, int numberOfActions) throws DynamicParamException {
        super (memory, numberOfStates, numberOfActions, false);
    }

    /**
     * Constructor for direct function estimator.
     *
     * @param memory memory reference.
     * @param numberOfStates number of states for direct function estimator
     * @param numberOfActions number of actions for direct function estimator
     * @param params parameters for function
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DirectFunctionEstimator(Memory memory, int numberOfStates, int numberOfActions, String params) throws DynamicParamException {
        super (memory, numberOfStates, numberOfActions, false, params);
    }

    /**
     * Returns reference to function estimator.
     *
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator reference() throws DynamicParamException {
        return new DirectFunctionEstimator(getMemory(), getNumberOfStates(), getNumberOfActions(), getParams());
    }

    /**
     * Returns reference to function estimator.
     *
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator reference(boolean sharedMemory) throws DynamicParamException {
        return new DirectFunctionEstimator(sharedMemory ? getMemory() : getMemory().reference(), getNumberOfStates(), getNumberOfActions(), getParams());
    }

    /**
     * Returns reference to function estimator.
     *
     * @param memory reference to memory.
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator reference(Memory memory) throws DynamicParamException {
        return new DirectFunctionEstimator(memory, getNumberOfStates(), getNumberOfActions(), getParams());
    }

    /**
     * Sets parameters used for direct function estimator.<br>
     *
     * @param params parameters used for direct function estimator.
     */
    public void setParams(DynamicParam params) {
    }

    /**
     * Not used.
     *
     */
    public void start() {
    }

    /**
     * Not used.
     *
     */
    public void stop() {
    }

    /**
     * Checks if function estimator is started.
     *
     * @return true if function estimator is started otherwise false.
     */
    public boolean isStarted() {
        return true;
    }

    /**
     * Returns shallow copy of direct function estimator.
     *
     * @return shallow copy of direct function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator copy() throws DynamicParamException {
        return new DirectFunctionEstimator(memory, getNumberOfStates(), getNumberOfActions(), getParams());
    }

    /**
     * Not used.
     *
     * @param stateTransition state
     * @return state value corresponding to a state
     */
    public Matrix predict(StateTransition stateTransition) {
        return null;
    }

    /**
     * Not used.

     * @param stateTransition state transition.
     * @param values values.
     */
    public void store(StateTransition stateTransition, Matrix values) {
    }

    /**
     * Not used.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void update() throws AgentException, MatrixException, NeuralNetworkException, IOException, DynamicParamException, ClassNotFoundException {
        updateComplete();
    }

    /**
     * Not used.
     *
     * @param functionEstimator estimator function used to update this function.
     * @param fullUpdate if true full update is done.
     */
    public void append(FunctionEstimator functionEstimator, boolean fullUpdate) {
    }

    /**
     * Appends parameters to this function estimator from another function estimator.
     *
     * @param functionEstimator function estimator used to update current function estimator.
     * @param tau tau which controls contribution of other function estimator.
     */
    public void append(FunctionEstimator functionEstimator, double tau) {
    }

    /**
     * Sets if importance sampling weights are applied.
     *
     * @param applyImportanceSamplingWeights if true importance sampling weights are applied otherwise not.
     */
    public void setEnableImportanceSamplingWeights(boolean applyImportanceSamplingWeights) {
    }

}
