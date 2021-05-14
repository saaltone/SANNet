/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.function;

import core.reinforcement.AgentException;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.StateTransition;
import utils.DynamicParam;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;

/**
 * Defines DirectFunctionEstimator (proxy for memory) to be used with PlainValueFunction.<br>
 *
 */
public class DirectFunctionEstimator extends AbstractFunctionEstimator {

    /**
     * Constructor for DirectFunctionEstimator.
     *
     * @param memory memory reference.
     * @param numberOfActions number of actions for DirectFunctionEstimator
     */
    public DirectFunctionEstimator(Memory memory, int numberOfActions) {
        super (memory, numberOfActions, false);
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
     */
    public FunctionEstimator copy() {
        return new DirectFunctionEstimator(memory, getNumberOfActions());
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
     * Returns parameters used for DirectFunctionEstimator.
     *
     * @return parameters used for DirectFunctionEstimator.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        return null;
    }

    /**
     * Sets parameters used for DirectFunctionEstimator.<br>
     *
     * @param params parameters used for OnlineMemory.
     */
    public void setParams(DynamicParam params) {
    }

}
