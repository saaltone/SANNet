/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.function;

import core.reinforcement.agent.State;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements direct function estimator (proxy for memory) to be used with plain value function.<br>
 *
 */
public class DirectFunctionEstimator extends AbstractFunctionEstimator {

    /**
     * Constructor for direct function estimator.
     *
     * @param functionEstimator function estimator reference.
     * @param params parameters for function
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public DirectFunctionEstimator(FunctionEstimator functionEstimator, String params) throws DynamicParamException, MatrixException {
        this (functionEstimator.getNumberOfStates(), functionEstimator.getNumberOfActions(), params);
    }

    /**
     * Constructor for direct function estimator.
     *
     * @param numberOfStates number of states for direct function estimator
     * @param numberOfActions number of actions for direct function estimator
     * @param params parameters for function
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public DirectFunctionEstimator(int numberOfStates, int numberOfActions, String params) throws DynamicParamException, MatrixException {
        super (numberOfStates, numberOfActions, false, params);
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
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public FunctionEstimator reference() throws DynamicParamException, MatrixException {
        return new DirectFunctionEstimator(getNumberOfStates(), getNumberOfActions(), getParams());
    }

    /**
     * Returns parameters used for abstract function estimator.
     *
     * @return parameters used for abstract function estimator.
     */
    public String getParamDefs() {
        return null;
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
     * Not used.
     *
     */
    protected void reset() {
    }

    /**
     * Returns shallow copy of direct function estimator.
     *
     * @return shallow copy of direct function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public FunctionEstimator copy() throws DynamicParamException, MatrixException {
        return new DirectFunctionEstimator(getNumberOfStates(), getNumberOfActions(), getParams());
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
     * Appends from function estimator.
     *
     * @param functionEstimator function estimator.
     */
    public void append(FunctionEstimator functionEstimator) {
    }

}
