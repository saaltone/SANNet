/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
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
     * Averaging tau.
     *
     */
    private final double averagingTau = 0.99;

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
     */
    AbstractValueFunction(String params) {
        initializeDefaultParams();
        this.params = params;
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
     * Updates value function for state set sampled from memory.
     *
     * @param sampledStates sampled states.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public void update(TreeSet<State> sampledStates) throws MatrixException, NeuralNetworkException, DynamicParamException {
        if (sampledStates == null) return;

        for (State state : sampledStates.descendingSet()) {
            state.tdTarget = state.reward + (state.isFinalState() ? 0 : gamma * getTargetValue(state.nextState));
            state.tdError = state.tdTarget - getStateValue(state);

            averageReward = averageReward == Double.MIN_VALUE ? state.reward : averagingTau * averageReward + (1 - averagingTau) * state.reward;
            averageTDTarget = averageTDTarget == Double.MIN_VALUE ? state.tdTarget : averagingTau * averageTDTarget + (1 - averagingTau) * state.tdTarget;
            averageTDError = averageTDError == Double.MIN_VALUE ? state.tdError : averagingTau * averageTDError + (1 - averagingTau) * state.tdError;
            if (tdDataPrintCycle > 0 && ++tdDataPrintCount >= tdDataPrintCycle) {
                System.out.println("Average Reward: " + averageReward + ", Average TD target: " + averageTDTarget + ", Average TD error: " + averageTDError);
                tdDataPrintCount = 0;
            }
        }

        updateBaseline(sampledStates);
    }

    /**
     * Return predicted state value.
     *
     * @param state state
     * @return predicted state value.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract double getStateValue(State state) throws NeuralNetworkException, MatrixException;

    /**
     * Returns target value based on next state.
     *
     * @param nextState next state.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    protected abstract double getTargetValue(State nextState) throws NeuralNetworkException, MatrixException, DynamicParamException;

    /**
     * Updates baseline value for states.
     *
     * @param states states.
     */
    protected abstract void updateBaseline(TreeSet<State> states);

}