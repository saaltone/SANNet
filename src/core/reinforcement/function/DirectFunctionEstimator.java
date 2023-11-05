/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.function;

import core.reinforcement.memory.Memory;
import core.reinforcement.agent.State;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;

/**
 * Implements direct function estimator (proxy for memory) to be used with plain value function.<br>
 *
 */
public class DirectFunctionEstimator extends AbstractFunctionEstimator {

    /**
     * Constructor for direct function estimator.
     *
     * @param functionEstimator function estimator reference.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DirectFunctionEstimator(FunctionEstimator functionEstimator) throws DynamicParamException {
        this (functionEstimator, null);
    }

    /**
     * Constructor for direct function estimator.
     *
     * @param functionEstimator function estimator reference.
     * @param params parameters for function
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DirectFunctionEstimator(FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        this (functionEstimator.getMemory(), functionEstimator.getNumberOfStates(), functionEstimator.getNumberOfActions(), params);
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
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
    }

    /**
     * Returns reference to function estimator.
     *
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator reference() throws DynamicParamException {
        return new DirectFunctionEstimator(this, getParams());
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
     * @param state state
     * @return policy values corresponding to a state
     */
    public Matrix predictPolicyValues(State state) {
        return null;
    }

    /**
     * Not used.
     *
     * @param state state.
     * @return policy values corresponding to a state.
     */
    public Matrix predictTargetPolicyValues(State state) {
        return null;
    }

    /**
     * Not used.
     *
     * @param state state
     * @return state action values corresponding to a state
     */
    public Matrix predictStateActionValues(State state) {
        return null;
    }

    /**
     * Not used.
     *
     * @param state state
     * @return state action values corresponding to a state
     */
    public Matrix predictTargetStateActionValues(State state) {
        return null;
    }

    /**
     * Not used.

     * @param state state.
     * @param values values.
     */
    public void storePolicyValues(State state, Matrix values) {
    }

    /**
     * Not used.
     *
     * @param state state.
     * @param values values.
     */
    public void storeStateActionValues(State state, Matrix values) {
    }

    /**
     * Not used.
     *
     */
    public void update() {
        updateComplete();
    }

    /**
     * Sets if importance sampling weights are applied.
     *
     * @param applyImportanceSamplingWeights if true importance sampling weights are applied otherwise not.
     */
    public void setEnableImportanceSamplingWeights(boolean applyImportanceSamplingWeights) {
    }

    /**
     * Appends from function estimator.
     *
     * @param functionEstimator function estimator.
     */
    public void append(FunctionEstimator functionEstimator) {
    }

}
