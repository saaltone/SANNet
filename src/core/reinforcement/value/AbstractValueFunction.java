/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.State;
import utils.configurable.Configurable;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.TreeSet;

/**
 * Implements abstract value function providing common functionality for value estimation.<br>
 *
 */
public abstract class AbstractValueFunction implements ValueFunction, Configurable, Serializable {

    @Serial
    private static final long serialVersionUID = -7436000520645598105L;

    /**
     * Parameter name types for abstract value function.
     *     - gamma: discount value for value function. Default value 0.99.<br>
     *     - tdDataPrintCycle: TD data print cycle. Default value 100.
     *
     */
    private final static String paramNameTypes = "(gamma:DOUBLE), " +
            "(tdDataPrintCycle:INT)";

    /**
     * Parameters for value function.
     *
     */
    private final String params;

    /**
     * Discount rate for temporal difference (TD) target calculation.
     *
     */
    private double gamma;

    /**
     * Moving average reward.
     *
     */
    private double averageReward = Double.MIN_VALUE;

    /**
     * Moving average TD target.
     *
     */
    private double averageTDTarget = Double.MIN_VALUE;

    /**
     * Moving average TD error.
     *
     */
    private double averageTDError = Double.MIN_VALUE;

    /**
     * Print cycle for average TD data verbosing.
     *
     */
    private int tdDataPrintCycle;

    /**
     * Count for average TD data verbosing.
     *
     */
    private transient int tdDataPrintCount = 0;

    /**
     * Constructor for abstract value function.
     *
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    AbstractValueFunction(String params) throws DynamicParamException {
        initializeDefaultParams();
        this.params = params;
        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        gamma = 0.99;
        tdDataPrintCycle = 100;
    }

    /**
     * Returns parameters of value function.
     *
     * @return parameters for value function.
     */
    protected String getParams() {
        return params;
    }

    /**
     * Returns parameters used for abstract value function.
     *
     * @return parameters used for abstract value function.
     */
    public String getParamDefs() {
        return AbstractValueFunction.paramNameTypes;
    }

    /**
     * Sets parameters used for abstract value function.<br>
     * <br>
     * Supported parameters are:<br>
     *     - gamma: discount value for value function. Default value 0.99.<br>
     *     - tdDataPrintCycle: TD data print cycle. Default value 100.
     *
     * @param params parameters used for abstract value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("gamma")) gamma = params.getValueAsDouble("gamma");
        if (params.hasParam("tdDataPrintCycle")) tdDataPrintCycle = params.getValueAsInteger("tdDataPrintCycle");
    }

    /**
     * Updates state value.
     *
     * @param state state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract void updateValue(State state) throws NeuralNetworkException, MatrixException;

    /**
     * Updates baseline value for states.
     *
     * @param states states.
     */
    protected abstract void updateBaseline(TreeSet<State> states);

    /**
     * Returns sampled states.
     *
     * @return sampled states.
     */
    protected abstract TreeSet<State> getSampledStates();

    /**
     * Updates value function for state set sampled from memory.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public void update() throws MatrixException, NeuralNetworkException {
        updateValue(getSampledStates());
    }

    /**
     * Updates values for current state chain end from of sequence to start.
     *
     * @param state state.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public void update(State state) throws NeuralNetworkException, MatrixException {
        TreeSet<State> states = new TreeSet<>();
        State currentState = state;
        while (currentState != null) {
            states.add(currentState);
            currentState = currentState.previousState;
        }
        updateValue(states);
    }

    /**
     * Updates values of states.
     *
     * @param states states.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    private void updateValue(TreeSet<State> states) throws MatrixException, NeuralNetworkException {
        if (states == null) return;

        for (State state : states.descendingSet()) {
            updateValue(state);
            double nextStateValue = state.isFinalState() ? 0 : gamma * getTargetValue(state.nextState);
            state.tdTarget = state.reward + nextStateValue;
            state.tdError = state.tdTarget - state.stateValue;
            state.advantage = state.policyValue - state.stateValue;
            averageReward = averageReward == Double.MIN_VALUE ? state.reward : 0.99 * averageReward + 0.01 * state.reward;
            averageTDTarget = averageTDTarget == Double.MIN_VALUE ? state.tdTarget : 0.99 * averageTDTarget + 0.01 * state.tdTarget;
            averageTDError = averageTDError == Double.MIN_VALUE ? state.tdError : 0.99 * averageTDError + 0.01 * state.tdError;
            if (tdDataPrintCycle > 0 && ++tdDataPrintCount >= tdDataPrintCycle) {
                System.out.println("Average Reward: " + averageReward + ", Average TD target: " + averageTDTarget + ", Average TD error: " + averageTDError);
                tdDataPrintCount = 0;
            }
        }

        updateBaseline(states);
    }

    /**
     * Returns target value based on next state.
     *
     * @param nextState next state.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract double getTargetValue(State nextState) throws NeuralNetworkException, MatrixException;

}