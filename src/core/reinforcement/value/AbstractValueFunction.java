/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.NeuralNetworkException;
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
     * If true value function is state value function otherwise action value function.
     *
     */
    protected final boolean isStateValue;

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
    protected double lambda = 1;

    /**
     * Moving average TDError.
     *
     */
    private double averageTDError = Double.NEGATIVE_INFINITY;

    /**
     * Print cycle for average TDError verbosing.
     *
     */
    private int tdErrorPrintCycle = 1000;

    /**
     * Count for average TDError verbosing.
     *
     */
    private int tdErrorPrintCount = 0;

    /**
     * Constructor for AbstractValueFunction.
     *
     */
    AbstractValueFunction() {
        this.numberOfActions = 1;
        isStateValue = true;
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
        isStateValue = false;
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
        paramDefs.put("tdErrorPrintCycle", DynamicParam.ParamType.INT);
        return paramDefs;
    }

    /**
     * Sets parameters used for AbstractValueFunction.<br>
     * <br>
     * Supported parameters are:<br>
     *     - gamma: discount value for value function. Default value 0.99.<br>
     *     - lambda: value controlling balance between bootstrapping and future reward of next state. Default value 1.<br>
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
    protected abstract double getValue(StateTransition stateTransition);

    /**
     * Updates state value.
     *
     * @param stateTransition state transition.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract void updateValue(StateTransition stateTransition) throws NeuralNetworkException, MatrixException;

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
     * Updates value function for set sampled from memory.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public void update() throws MatrixException, NeuralNetworkException {
        if (getFunctionEstimator().sampledSetEmpty()) return;
        updateValue(getFunctionEstimator().getSampledStateTransitions());
    }

    /**
     * Updates values for current episode.
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