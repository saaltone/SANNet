/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.State;
import core.reinforcement.function.FunctionEstimator;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.TreeSet;

/**
 * Implements abstract value function estimator providing common functions for value function estimators.<br>
 *
 */
public abstract class AbstractValueFunctionEstimator extends AbstractValueFunction {

    /**
     * Reference to function estimator.
     *
     */
    protected final FunctionEstimator functionEstimator;

    /**
     * If true function estimator is state action value function.
     *
     */
    private final boolean isStateActionValueFunction;

    /**
     * Constructor for abstract value function estimator
     *
     * @param functionEstimator reference to function estimator.
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractValueFunctionEstimator(FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super(params);
        this.functionEstimator = functionEstimator;
        this.isStateActionValueFunction = getFunctionEstimator().isStateActionValueFunction();
        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for abstract value function estimator.
     *
     * @return parameters used for abstract value function estimator.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + getFunctionEstimator().getParamDefs();
    }

    /**
     * Sets parameters used for abstract value function estimator.<br>
     *
     * @param params parameters used for abstract value function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        getFunctionEstimator().setParams(params);
    }

    /**
     * Starts function estimator
     *
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start(Agent agent) throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException {
        if (!isStateActionValueFunction()) getFunctionEstimator().registerAgent(agent);
        getFunctionEstimator().start();
    }

    /**
     * Stops function estimator
     *
     * @throws NeuralNetworkException throws exception is neural network is not started.
     */
    public void stop() throws NeuralNetworkException {
        getFunctionEstimator().stop();
    }

    /**
     * Return true is function is state action value function.
     *
     * @return true is function is state action value function.
     */
    public boolean isStateActionValueFunction() {
        return isStateActionValueFunction;
    }

    /**
     * Returns value function index.
     *
     * @param state state.
     * @return value function index.
     */
    protected abstract int getValueFunctionIndex(State state);

    /**
     * Updates state value.
     *
     * @param state state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void updateValue(State state) throws NeuralNetworkException, MatrixException {
        state.stateValue = getFunctionEstimator().predictStateActionValues(state).getValue(getValueFunctionIndex(state), 0, 0);
    }

    /**
     * Updates baseline value for states.
     *
     * @param states states.
     */
    protected void updateBaseline(TreeSet<State> states) {
    }

    /**
     * Resets function estimator.
     *
     */
    public void resetFunctionEstimator() {
        getFunctionEstimator().reset();
    }

    /**
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @throws AgentException throws exception if agent is not registered for function estimator.
     * @return true if all registered agents are ready to update.
     */
    public boolean readyToUpdate(Agent agent) throws AgentException {
        return isStateActionValueFunction() || getFunctionEstimator().readyToUpdate(agent);
    }

    /**
     * Samples memory of function estimator.
     *
     */
    public void sample() {
        getFunctionEstimator().sample();
    }

    /**
     * Returns sampled states.
     *
     * @return sampled states.
     */
    public TreeSet<State> getSampledStates() {
        return getFunctionEstimator().getSampledStates();
    }

    /**
     * Returns function estimator.
     *
     * @return function estimator.
     */
    public FunctionEstimator getFunctionEstimator() {
        return functionEstimator;
    }

    /**
     * Updates function estimator.
     *
     * @return sampled states.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public TreeSet<State> updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException {
        TreeSet<State> sampledStates = getSampledStates();
        if (sampledStates == null || sampledStates.isEmpty()) getFunctionEstimator().abortUpdate();
        else {
            getFunctionEstimator().update(sampledStates);
            updateTargetValues(getFunctionEstimator(), sampledStates);
            if (!isStateActionValueFunction()) getFunctionEstimator().update();
        }
        return sampledStates;
    }

    /**
     * Updates target values into function estimator.
     *
     * @param currentFunctionEstimator current function estimator.
     * @param sampledStates sampled states.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     */
    protected void updateTargetValues(FunctionEstimator currentFunctionEstimator, TreeSet<State> sampledStates) throws MatrixException, NeuralNetworkException {
        for (State state : sampledStates) {
            Matrix targetValues = currentFunctionEstimator.predictStateActionValues(state);
            targetValues.setValue(getValueFunctionIndex(state), 0, 0, state.tdTarget);
            currentFunctionEstimator.storeStateActionValues(state, targetValues);
        }
    }

}
