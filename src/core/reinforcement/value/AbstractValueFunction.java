/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.NeuralNetworkException;
import core.reinforcement.*;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashMap;
import java.util.TreeMap;

/**
 * Class that defines AbstractValueFunction.
 *
 */
public abstract class AbstractValueFunction implements ValueFunction, Serializable {

    private static final long serialVersionUID = -7436000520645598105L;

    /**
     * Current episode count.
     *
     */
    protected transient int episodeCount;

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
     * If true uses baseline for target value update.
     *
     */
    private boolean useBaseline = false;

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
        setParams(new DynamicParam(params, getParamDefs()));
        this.numberOfActions = numberOfActions;
    }

    /**
     * Returns parameters used for AbstractValueFunction.
     *
     * @return parameters used for AbstractValueFunction.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("gamma", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("useBaseline", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for AbstractValueFunction.<br>
     * <br>
     * Supported parameters are:<br>
     *     - size: discount (gamma) value for value function. Default value 0.99.<br>
     *     - useBaseline: if true uses baseline (advantage) for value function. Default value false.<br>
     *
     * @param params parameters used for AbstractValueFunction.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("gamma")) gamma = params.getValueAsDouble("gamma");
        if (params.hasParam("useBaseline")) useBaseline = params.getValueAsBoolean("useBaseline");
    }

    /**
     * Sets current episode count.
     *
     * @param episodeCount current episode count.
     */
    public void setEpisode(int episodeCount) {
        this.episodeCount = episodeCount;
    }

    /**
     * Checks if baseline is in use.
     *
     * @return returns true if baseline is used.
     */
    protected boolean useBaseline() {
        return useBaseline;
    }

    /**
     * Returns values for state.
     *
     * @param state state.
     * @return values for state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract Matrix getValues(State state) throws NeuralNetworkException, MatrixException;

    /**
     * Returns number of actions.
     *
     * @return number of actions.
     */
    protected int getNumberOfActions() {
        return numberOfActions;
    }

    /**
     * Returns current action of state.
     *
     * @param state state.
     * @return current action of sample.
     */
    protected int getAction(State state) {
        return numberOfActions == 1 ? 0 : state.action;
    }

    /**
     * Returns TD target of sample.
     *
     * @param state state.
     * @return TD target of state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getTDTarget(State state) throws NeuralNetworkException, MatrixException {
        return state.reward + gamma * (state.isFinalState() ? 0 : getTargetValue(state.nextState));
    }

    /**
     * Updates TD target of sample.
     *
     * @param sample sample.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void updateTDTarget(RLSample sample) throws NeuralNetworkException, MatrixException {
        sample.stateValues = getValues(sample.state);
        if (useBaseline()) updateBaseline(sample);
        sample.tdTarget = getTDTarget(sample.state);
        sample.tdError = sample.tdTarget - sample.getValue(getAction(sample.state));
        sample.setValue(getAction(sample.state), sample.tdTarget);
        if (!sample.state.isFinalState()) sample.timeStep = sample.state.nextState.sample.timeStep;
    }

    /**
     * Updates baseline value for sample.
     *
     * @param sample sample.
     */
    protected abstract void updateBaseline(RLSample sample);

    /**
     * Updates function estimation using samples with updated TD targets.
     *
     * @param samples samples for update.
     * @param hasImportanceSamplingWeights if true samples contain importance sampling weights otherwise false.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void updateFunctionEstimator(TreeMap<Integer, RLSample> samples, boolean hasImportanceSamplingWeights) throws NeuralNetworkException, MatrixException, DynamicParamException {
        for (Integer sampleIndex : samples.descendingKeySet()) updateTDTarget(samples.get(sampleIndex));
    }

}