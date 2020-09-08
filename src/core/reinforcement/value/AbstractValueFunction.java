/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.NeuralNetworkException;
import core.reinforcement.Agent;
import core.reinforcement.AgentException;
import core.reinforcement.memory.StateTransition;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashMap;
import java.util.TreeSet;

/**
 * Class that defines AbstractValueFunction.
 *
 */
public abstract class AbstractValueFunction implements ValueFunction, Serializable {

    private static final long serialVersionUID = -7436000520645598105L;

    /**
     * Number of actions for value function.
     *
     */
    private final int numberOfActions;

    /**
     * Discount rate for temporal difference (TD) target calculation.
     *
     */
    private double gamma = 0.99;

    /**
     * Lambda value controlling balance between bootstrapped value and future reward of next state.
     *
     */
    protected double lambda = 0.75;

    /**
     * Constructor for AbstractValueFunction.
     *
     */
    AbstractValueFunction() {
        this.numberOfActions = 1;
    }

    /**
     * Constructor for AbstractValueFunction.
     *
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    AbstractValueFunction(String params) throws DynamicParamException {
        this();
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Constructor for AbstractValueFunction.
     *
     * @param numberOfActions number of actions for AbstractValueFunction.
     */
    AbstractValueFunction(int numberOfActions) {
        this.numberOfActions = numberOfActions;
    }

    /**
     * Constructor for AbstractValueFunction.
     *
     * @param numberOfActions number of actions for AbstractValueFunction.
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    AbstractValueFunction(int numberOfActions, String params) throws DynamicParamException {
        this(numberOfActions);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for AbstractValueFunction.
     *
     * @return parameters used for AbstractValueFunction.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("gamma", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("lambda", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for AbstractValueFunction.<br>
     * <br>
     * Supported parameters are:<br>
     *     - gamma: discount value for value function. Default value 0.99.<br>
     *     - lambda: value controlling balance between bootstrapping and future reward of next state. Default value 0.75.<br>
     *
     * @param params parameters used for AbstractValueFunction.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("gamma")) gamma = params.getValueAsDouble("gamma");
        if (params.hasParam("lambda")) lambda = params.getValueAsDouble("lambda");
    }

    /**
     * Returns number of actions.
     *
     * @return number of actions.
     */
    protected int getNumberOfActions() {
        return numberOfActions;
    }

    /**
     * Returns value for state.
     *
     * @param stateTransition state transition.
     * @return value for state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract double getValue(StateTransition stateTransition) throws NeuralNetworkException, MatrixException;

    /**
     * Returns target value based on next state.
     *
     * @param nextStateTransition next state transition.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract double getTargetValue(StateTransition nextStateTransition) throws NeuralNetworkException, MatrixException;

    /**
     * Updates baseline value for state transitions.
     *
     * @param stateTransitions state transitions.
     */
    protected abstract void updateBaseline(TreeSet<StateTransition> stateTransitions);

    /**
     * Updates value function.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    public void update(Agent agent) throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException {
        getFunctionEstimator().sample();
        if (getFunctionEstimator().sampledSetEmpty()) return;
        TreeSet<StateTransition> stateTransitions = getFunctionEstimator().getSampledStateTransitions();

        for (StateTransition stateTransition : stateTransitions) stateTransition.stateValue = getValue(stateTransition);
        for (StateTransition stateTransition : stateTransitions.descendingSet()) {
            stateTransition.tdTarget = stateTransition.reward + (stateTransition.isFinalState() ? 0 : gamma * ((1 - lambda) * getTargetValue(stateTransition.nextStateTransition) + lambda * stateTransition.nextStateTransition.tdTarget));
            stateTransition.tdError = stateTransition.tdTarget - stateTransition.stateValue;
        }

        updateBaseline(stateTransitions);

        updateFunctionEstimator(agent, stateTransitions);
    }

    /**
     * Updates FunctionEstimator.
     *
     * @param agent agent.
     * @param stateTransitions state transitions used to update FunctionEstimator.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    protected abstract void updateFunctionEstimator(Agent agent, TreeSet<StateTransition> stateTransitions) throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException;

}