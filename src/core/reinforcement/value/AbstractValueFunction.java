/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.StateTransition;
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
     *     - lambda: value controlling balance between bootstrapping and future reward of next state. Default value 1.<br>
     *     - tdDataPrintCycle: TD data print cycle. Default value 100.
     *
     */
    private final static String paramNameTypes = "(gamma:DOUBLE), " +
            "(lambda:DOUBLE), " +
            "(tdDataPrintCycle:INT)";

    /**
     * Number of actions for value function.
     *
     */
    private final int numberOfActions;

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
     * Lambda value controlling balance between bootstrapped value and future reward of next state.
     *
     */
    protected double lambda;

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
    private int tdDataPrintCount;

    /**
     * Constructor for abstract value function.
     *
     * @param numberOfActions number of actions for abstract value function.
     */
    AbstractValueFunction(int numberOfActions) {
        initializeDefaultParams();
        this.numberOfActions = numberOfActions;
        this.params = null;
    }

    /**
     * Constructor for abstract value function.
     *
     * @param numberOfActions number of actions for abstract value function.
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    AbstractValueFunction(int numberOfActions, String params) throws DynamicParamException {
        initializeDefaultParams();
        this.numberOfActions = numberOfActions;
        this.params = params;
        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        gamma = 0.99;
        lambda = 1;
        tdDataPrintCycle = 100;
        tdDataPrintCount = 0;
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
     *     - lambda: value controlling balance between bootstrapping and future reward of next state. Default value 0.<br>
     *     - tdDataPrintCycle: TD data print cycle. Default value 100.
     *
     * @param params parameters used for abstract value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("gamma")) gamma = params.getValueAsDouble("gamma");
        if (params.hasParam("lambda")) lambda = params.getValueAsDouble("lambda");
        if (params.hasParam("tdDataPrintCycle")) tdDataPrintCycle = params.getValueAsInteger("tdDataPrintCycle");
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
     */
    private double getValue(StateTransition stateTransition) {
        return stateTransition.value;
    }

    /**
     * Updates state value.
     *
     * @param stateTransition state transition.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract void updateValue(StateTransition stateTransition) throws NeuralNetworkException, MatrixException;

    /**
     * Updates baseline value for state transitions.
     *
     * @param stateTransitions state transitions.
     */
    protected abstract void updateBaseline(TreeSet<StateTransition> stateTransitions);

    /**
     * Returns sampled state transitions.
     *
     * @return sampled state transitions.
     */
    protected abstract TreeSet<StateTransition> getSampledStateTransitions();

    /**
     * Updates value function for set sampled from memory.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public void update() throws MatrixException, NeuralNetworkException {
        updateValue(getSampledStateTransitions());
    }

    /**
     * Updates values for current state transition chain.
     *
     * @param stateTransition state transition.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public void update(StateTransition stateTransition) throws NeuralNetworkException, MatrixException {
        StateTransition currentStateTransition = stateTransition;
        TreeSet<StateTransition> stateTransitions = new TreeSet<>();
        while (currentStateTransition != null) {
            stateTransitions.add(currentStateTransition);
            currentStateTransition = currentStateTransition.previousStateTransition;
        }
        updateValue(stateTransitions);
    }

    /**
     * Updates value of state transitions.
     *
     * @param stateTransitions state transitions.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    private void updateValue(TreeSet<StateTransition> stateTransitions) throws MatrixException, NeuralNetworkException {
        if (stateTransitions == null) return;

        for (StateTransition stateTransition : stateTransitions.descendingSet()) {
            updateValue(stateTransition);
            stateTransition.tdTarget = stateTransition.reward + (stateTransition.isFinalState() ? 0 : gamma * ((lambda == 0 ? getValue(stateTransition.nextStateTransition) : lambda == 1 ? getTargetValue(stateTransition.nextStateTransition) : (1 - lambda) * getValue(stateTransition.nextStateTransition) + lambda * getTargetValue(stateTransition.nextStateTransition))));
            stateTransition.tdError = stateTransition.tdTarget - getValue(stateTransition);
            stateTransition.advantage = stateTransition.tdError;
            averageReward = averageReward == Double.MIN_VALUE ? stateTransition.reward : 0.99 * averageReward + 0.01 * stateTransition.reward;
            averageTDTarget = averageTDTarget == Double.MIN_VALUE ? stateTransition.tdTarget : 0.99 * averageTDTarget + 0.01 * stateTransition.tdTarget;
            averageTDError = averageTDError == Double.MIN_VALUE ? stateTransition.tdError : 0.99 * averageTDError + 0.01 * stateTransition.tdError;
            if (tdDataPrintCycle > 0 && ++tdDataPrintCount % tdDataPrintCycle == 0) {
                System.out.println("Average Reward: " + averageReward + ", Average TD target: " + averageTDTarget + ", Average TD error: " + averageTDError);
            }
        }

        updateBaseline(stateTransitions);
    }

    /**
     * Returns target value based on next state.
     *
     * @param nextStateTransition next state transition.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract double getTargetValue(StateTransition nextStateTransition) throws NeuralNetworkException, MatrixException;

}