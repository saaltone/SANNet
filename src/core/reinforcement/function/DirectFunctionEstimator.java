/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.function;

import core.reinforcement.agent.AgentException;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.StateTransition;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Defines DirectFunctionEstimator (proxy for memory) to be used with PlainValueFunction.<br>
 *
 */
public class DirectFunctionEstimator extends AbstractFunctionEstimator {

    /**
     * Constructor for DirectFunctionEstimator.
     *
     * @param memory memory reference.
     * @param numberOfStates number of states for TabularFunctionEstimator
     * @param numberOfActions number of actions for DirectFunctionEstimator
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DirectFunctionEstimator(Memory memory, int numberOfStates, int numberOfActions) throws DynamicParamException {
        super (memory, numberOfStates, numberOfActions, false);
    }

    /**
     * Constructor for DirectFunctionEstimator.
     *
     * @param memory memory reference.
     * @param numberOfStates number of states for TabularFunctionEstimator
     * @param numberOfActions number of actions for DirectFunctionEstimator
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
     * Returns shallow copy of DirectFunctionEstimator.
     *
     * @return shallow copy of DirectFunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator copy() throws DynamicParamException {
        return new DirectFunctionEstimator(memory, getNumberOfStates(), getNumberOfActions(), getParams());
    }

    /**
     * Not used.
     *
     * @param state state
     * @return state value corresponding to a state
     */
    public Matrix predict(Matrix state) {
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
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void update() throws AgentException, MatrixException {
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
     * Sets parameters used for DirectFunctionEstimator.<br>
     *
     * @param params parameters used for OnlineMemory.
     */
    public void setParams(DynamicParam params) {
    }

}
