/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.memory.StateTransition;
import utils.Configurable;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.TreeSet;

/**
 * Class that defines AbstractValueFunction.<br>
 *
 */
public abstract class AbstractValueFunction implements ValueFunction, Configurable, Serializable {

    @Serial
    private static final long serialVersionUID = -7436000520645598105L;

    /**
     * Parameter name types for AbstractValueFunction.
     *     - gamma: discount value for value function. Default value 0.99.<br>
     *     - lambda: value controlling balance between bootstrapping and future reward of next state. Default value 1.<br>
     *     - tdErrorPrintCycle: TD error print cycle. Default value 1000.
     *
     */
    private final static String paramNameTypes = "(gamma:DOUBLE), " +
            "(lambda:DOUBLE), " +
            "(tdErrorPrintCycle:INT)";

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
     * Moving average TDError.
     *
     */
    private double averageTDError = Double.NEGATIVE_INFINITY;

    /**
     * Print cycle for average TDError verbosing.
     *
     */
    private int tdErrorPrintCycle;

    /**
     * Count for average TDError verbosing.
     *
     */
    private int tdErrorPrintCount = 0;

    /**
     * Constructor for AbstractValueFunction.
     *
     * @param numberOfActions number of actions for AbstractValueFunction.
     */
    AbstractValueFunction(int numberOfActions) {
        initializeDefaultParams();
        this.numberOfActions = numberOfActions;
        this.params = null;
    }

    /**
     * Constructor for AbstractValueFunction.
     *
     */
    AbstractValueFunction() {
        initializeDefaultParams();
        this.numberOfActions = 1;
        this.params = null;
    }

    /**
     * Constructor for AbstractValueFunction.
     *
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    AbstractValueFunction(String params) throws DynamicParamException {
        initializeDefaultParams();
        this.numberOfActions = 1;
        this.params = params;
        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Constructor for AbstractValueFunction.
     *
     * @param numberOfActions number of actions for AbstractValueFunction.
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
        lambda = 0;
        tdErrorPrintCycle = 1000;
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
     * Returns parameters used for AbstractValueFunction.
     *
     * @return parameters used for AbstractValueFunction.
     */
    public String getParamDefs() {
        return AbstractValueFunction.paramNameTypes;
    }

    /**
     * Sets parameters used for AbstractValueFunction.<br>
     * <br>
     * Supported parameters are:<br>
     *     - gamma: discount value for value function. Default value 0.99.<br>
     *     - lambda: value controlling balance between bootstrapping and future reward of next state. Default value 0.<br>
     *     - tdErrorPrintCycle: TD error print cycle. Default value 1000.
     *
     * @param params parameters used for AbstractValueFunction.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("gamma")) gamma = params.getValueAsDouble("gamma");
        if (params.hasParam("lambda")) lambda = params.getValueAsDouble("lambda");
        if (params.hasParam("tdErrorPrintCycle")) tdErrorPrintCycle = params.getValueAsInteger("tdErrorPrintCycle");
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
    public double getValue(StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
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
            stateTransition.tdTarget = stateTransition.reward + (stateTransition.isFinalState() ? 0 : gamma * ((1 - lambda) * getValue(stateTransition.nextStateTransition) + lambda * getTargetValue(stateTransition.nextStateTransition)));
            stateTransition.tdError = stateTransition.tdTarget - getValue(stateTransition);
            stateTransition.advantage = stateTransition.tdError;
            averageTDError = averageTDError == Double.NEGATIVE_INFINITY ? stateTransition.tdError : 0.99 * averageTDError + 0.01 * stateTransition.tdError;
            if (tdErrorPrintCycle > 0 && ++tdErrorPrintCount % tdErrorPrintCycle == 0) System.out.println("Average TD error: " + averageTDError);
        }

        updateBaseline(stateTransitions);
    }

}