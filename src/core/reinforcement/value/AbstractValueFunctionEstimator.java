/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.NeuralNetworkException;
import core.reinforcement.Agent;
import core.reinforcement.AgentException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;
import java.util.TreeSet;

/**
 * Class that defines AbstractValueFunctionEstimator.
 *
 */
public abstract class AbstractValueFunctionEstimator extends AbstractValueFunction {

    /**
     * Reference to FunctionEstimator.
     *
     */
    protected final FunctionEstimator functionEstimator;

    /**
     * Constructor for AbstractValueFunctionEstimator
     *
     * @param numberOfActions number of actions for AbstractValueFunctionEstimator.
     * @param functionEstimator reference to FunctionEstimator.
     */
    public AbstractValueFunctionEstimator(int numberOfActions, FunctionEstimator functionEstimator) {
        super(numberOfActions);
        this.functionEstimator = functionEstimator;
    }

    /**
     * Constructor for AbstractValueFunctionEstimator
     *
     * @param numberOfActions number of actions for AbstractValueFunctionEstimator.
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractValueFunctionEstimator(int numberOfActions, FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        this(numberOfActions, functionEstimator);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for AbstractValueFunctionEstimator.
     *
     * @return parameters used for AbstractValueFunctionEstimator.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.putAll(functionEstimator.getParamDefs());
        return paramDefs;
    }

    /**
     * Sets parameters used for AbstractValueFunctionEstimator.<br>
     *
     * @param params parameters used for AbstractValueFunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        functionEstimator.setParams(params);
    }

    /**
     * Starts FunctionEstimator
     *
     * @throws NeuralNetworkException throws exception if starting of value FunctionEstimator fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException {
        functionEstimator.start();
    }

    /**
     * Stops FunctionEstimator
     *
     */
    public void stop() {
        functionEstimator.stop();
    }

    /**
     * Registers agent for FunctionEstimator.
     *
     * @param agent agent.
     */
    public void registerAgent(Agent agent) {
        if (!isStateActionValueFunction()) functionEstimator.registerAgent(agent);
    }

    /**
     * Return true is function is state action value function.
     *
     * @return true is function is state action value function.
     */
    public boolean isStateActionValueFunction() {
        return functionEstimator.isStateActionValueFunction();
    }

    /**
     * Returns action with potential state action value offset.
     *
     * @param action action.
     * @return updated action.
     */
    protected int getAction(int action) {
        return (isStateActionValueFunction() ? 1 : 0) + action;
    }

    /**
     * Returns values for state.
     *
     * @param stateTransition state transition.
     * @return values for state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix getValues(StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
        return functionEstimator.predict(stateTransition.environmentState.state);
    }

    /**
     * Updates state value.
     *
     * @param stateTransition state transition.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void updateValue(StateTransition stateTransition) throws NeuralNetworkException, MatrixException {
        stateTransition.value = getValues(stateTransition).getValue(getAction(stateTransition.action), 0);
    }

    /**
     * Updates baseline value for state transitions.
     *
     * @param stateTransitions state transitions.
     */
    protected void updateBaseline(TreeSet<StateTransition> stateTransitions) {
    }

    /**
     * Resets FunctionEstimator.
     *
     */
    public void resetFunctionEstimator() {
        functionEstimator.reset();
    }

    /**
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @throws AgentException throws exception if agent is not registered for function estimator.
     * @return true if all registered agents are ready to update.
     */
    public boolean readyToUpdate(Agent agent) throws AgentException {
        return functionEstimator.readyToUpdate(agent);
    }

    /**
     * Updated state transitions in memory of FunctionEstimator.
     *
     * @param stateTransitions state transitions
     */
    public void updateFunctionEstimatorMemory(TreeSet<StateTransition> stateTransitions) {
        functionEstimator.update(stateTransitions);
    }

    /**
     * Samples memory of FunctionEstimator.
     *
     */
    public void sample() {
        functionEstimator.sample();
    }

    /**
     * Returns sampled state transitions.
     *
     * @return sampled state transitions.
     */
    public TreeSet<StateTransition> getSampledStateTransitions() {
        return functionEstimator.getSampledStateTransitions();
    }

    /**
     * Updates FunctionEstimator.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    public void updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException {
        TreeSet<StateTransition> sampledStateTransitions = functionEstimator.getSampledStateTransitions();
        if (sampledStateTransitions == null || sampledStateTransitions.isEmpty()) return;

        updateFunctionEstimatorMemory(sampledStateTransitions);

        if (!isStateActionValueFunction()) {
            for (StateTransition stateTransition : sampledStateTransitions) functionEstimator.store(stateTransition, getTargetValues(stateTransition));
            functionEstimator.update();
        }
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
        Matrix targetValues = getValues(stateTransition).copy();
        targetValues.setValue(getAction(stateTransition.action), 0, stateTransition.tdTarget);
        return targetValues;
    }

}
