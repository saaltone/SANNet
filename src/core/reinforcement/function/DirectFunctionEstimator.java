/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.function;

import core.reinforcement.Agent;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.StateTransition;
import utils.matrix.Matrix;

/**
 * Defines DirectFunctionEstimator (proxy for memory) to be used with PlainValueFunction.
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
        super (memory, numberOfActions);
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
     * Not used.
     *
     * @param agent agent.
     */
    public void registerAgent(Agent agent) {
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
     *
     * @param agent agent.
     * @param stateTransition state transition.
     * @param values values.
     */
    public void store(Agent agent, StateTransition stateTransition, Matrix values) {
    }

    /**
     * Not used.
     *
     * @param agent agent.
     */
    public void update(Agent agent) {
    }

    /**
     * Not used.
     *
     * @param functionEstimator estimator function used to update this function.
     * @param fullUpdate if true full update is done.
     */
    public void append(FunctionEstimator functionEstimator, boolean fullUpdate) {
    }

}
