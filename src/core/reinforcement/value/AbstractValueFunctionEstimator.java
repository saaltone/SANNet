/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.StateTransition;
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
     * @param numberOfActions number of actions for abstract value function estimator.
     * @param functionEstimator reference to function estimator.
     */
    public AbstractValueFunctionEstimator(int numberOfActions, FunctionEstimator functionEstimator) {
        super(numberOfActions);
        this.functionEstimator = functionEstimator;
        this.isStateActionValueFunction = getFunctionEstimator().isStateActionValueFunction();
    }

    /**
     * Constructor for abstract value function estimator
     *
     * @param numberOfActions number of actions for abstract value function estimator.
     * @param functionEstimator reference to function estimator.
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractValueFunctionEstimator(int numberOfActions, FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super(numberOfActions, params);
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
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException {
        getFunctionEstimator().start();
    }

    /**
     * Stops function estimator
     *
     */
    public void stop() {
        getFunctionEstimator().stop();
    }

    /**
     * Registers agent for function estimator.
     *
     * @param agent agent.
     */
    public void registerAgent(Agent agent) {
        if (!isStateActionValueFunction()) getFunctionEstimator().registerAgent(agent);
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
     * @param stateTransition state transition.
     * @return value function index.
     */
    protected abstract int getValueFunctionIndex(StateTransition stateTransition);

    /**
     * Returns values for state.
     *
     * @param currentFunctionEstimator current function estimator.
     * @param stateTransition state.
     * @return values for state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix getValues(FunctionEstimator currentFunctionEstimator, StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
        return currentFunctionEstimator.predictStateActionValues(stateTransition);
    }

    /**
     * Updates state value.
     *
     * @param stateTransition state transition.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void updateValue(StateTransition stateTransition) throws NeuralNetworkException, MatrixException {
        stateTransition.value = getValues(getFunctionEstimator(), stateTransition).getValue(getValueFunctionIndex(stateTransition), 0, 0);
    }

    /**
     * Updates baseline value for state transitions.
     *
     * @param stateTransitions state transitions.
     */
    protected void updateBaseline(TreeSet<StateTransition> stateTransitions) {
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
     * Updates state transitions in memory of FunctionEstimator.
     *
     * @param stateTransitions state transitions
     */
    public void updateFunctionEstimatorMemory(TreeSet<StateTransition> stateTransitions) {
        getFunctionEstimator().update(stateTransitions);
    }

    /**
     * Samples memory of function estimator.
     *
     */
    public void sample() {
        getFunctionEstimator().sample();
    }

    /**
     * Returns sampled state transitions.
     *
     * @return sampled state transitions.
     */
    public TreeSet<StateTransition> getSampledStateTransitions() {
        return getFunctionEstimator().getSampledStateTransitions();
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
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException, IOException, ClassNotFoundException {
        TreeSet<StateTransition> sampledStateTransitions = getFunctionEstimator().getSampledStateTransitions();
        if (sampledStateTransitions == null || sampledStateTransitions.isEmpty()) {
            getFunctionEstimator().abortUpdate();
            return;
        }

        updateFunctionEstimatorMemory(sampledStateTransitions);

        for (StateTransition stateTransition : sampledStateTransitions) getFunctionEstimator().storeStateActionValues(stateTransition, getTargetValues(stateTransition));
        if (!isStateActionValueFunction()) getFunctionEstimator().update();
    }

    /**
     * Returns target values.
     *
     * @param stateTransition state transition.
     * @return target values.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix getTargetValues(StateTransition stateTransition) throws NeuralNetworkException, MatrixException {
        Matrix targetValues = getValues(getFunctionEstimator(), stateTransition).copy();
        targetValues.setValue(getValueFunctionIndex(stateTransition), 0, 0, stateTransition.tdTarget);
        return targetValues;
    }

    /**
     * Appends parameters to this value function from another value function.
     *
     * @param valueFunction value function used to update current value function.
     * @param tau tau which controls contribution of other value function.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void append(ValueFunction valueFunction, double tau) throws MatrixException, AgentException, NeuralNetworkException, IOException, DynamicParamException, ClassNotFoundException {
        getFunctionEstimator().append(valueFunction.getFunctionEstimator(), tau);
    }

}
