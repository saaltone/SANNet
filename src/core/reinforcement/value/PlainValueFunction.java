/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.reinforcement.Agent;
import core.reinforcement.function.DirectFunctionEstimator;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import utils.DynamicParam;
import utils.DynamicParamException;

import java.util.HashMap;
import java.util.TreeSet;

/**
 * Class that defines PlainValueFunction without function estimator.
 *
 */
public class PlainValueFunction extends AbstractValueFunction {

    private final DirectFunctionEstimator functionEstimator;

    /**
     * If true uses baseline for target value update.
     *
     */
    private boolean useBaseline = true;

    /**
     * Tau value for baseline (mean and standard deviation) averaging.
     *
     */
    private double tau = 0.9;

    /**
     * Average mean.
     *
     */
    private double averageMean = Double.NEGATIVE_INFINITY;

    /**
     * Average standard deviation.
     *
     */
    private double averageStd = Double.NEGATIVE_INFINITY;

    /**
     * Constructor for PlainValueFunction.
     *
     * @param functionEstimator reference to DirectFunctionEstimator.
     */
    public PlainValueFunction(DirectFunctionEstimator functionEstimator) {
        super(1);
        this.functionEstimator = functionEstimator;
    }

    /**
     * Constructor for PlainValueFunction.
     *
     * @param numberOfActions number of actions for PlainValueFunction.
     * @param functionEstimator reference to DirectFunctionEstimator.
     */
    public PlainValueFunction(int numberOfActions, DirectFunctionEstimator functionEstimator) {
        super(numberOfActions);
        this.functionEstimator = functionEstimator;
        lambda = 1;
    }

    /**
     * Constructor for PlainValueFunction.
     *
     * @param functionEstimator reference to DirectFunctionEstimator.
     * @param params parameters for PlainValueFunction.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public PlainValueFunction(DirectFunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super(1, params);
        this.functionEstimator = functionEstimator;
        lambda = 1;
    }

    /**
     * Constructor for PlainValueFunction.
     *
     * @param numberOfActions number of actions for PlainValueFunction.
     * @param functionEstimator reference to DirectFunctionEstimator.
     * @param params parameters for PlainValueFunction.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public PlainValueFunction(int numberOfActions, DirectFunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super(numberOfActions, params);
        this.functionEstimator = functionEstimator;
        lambda = 1;
    }

    /**
     * Returns parameters used for PlainValueFunction.
     *
     * @return parameters used for PlainValueFunction.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.put("useBaseline", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("tau", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for PlainValueFunction.<br>
     * <br>
     * Supported parameters are:<br>
     *     - useBaseline: if true uses baseline for value function. Default value true.<br>
     *     - tau: tau value for baseline (mean and standard deviation) averaging. Default value 0.9.<br>
     *
     * @param params parameters used for PlainValueFunction.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("useBaseline")) useBaseline = params.getValueAsBoolean("useBaseline");
        if (params.hasParam("tau")) tau = params.getValueAsDouble("tau");
    }

    /**
     * Not used.
     *
     */
    public void start() {}

    /**
     * Not used.
     *
     */
    public void stop() {}

    /**
     * Returns state value.
     *
     * @param stateTransition state.
     * @return state value.
     */
    protected double getValue(StateTransition stateTransition) {
        return 0;
    }

    /**
     * Updates state value.
     *
     * @param stateTransition state transition.
     */
    protected void updateValue(StateTransition stateTransition) {
    }

    /**
     * Returns target value based on next state.
     *
     * @param nextStateTransition next state transition.
     * @return target value based on next state
     */
    public double getTargetValue(StateTransition nextStateTransition) {
        return nextStateTransition.tdTarget;
    }

    /**
     * Updates baseline values for state transitions.
     *
     * @param stateTransitions state transitions.
     */
    protected void updateBaseline(TreeSet<StateTransition> stateTransitions) {
        if (!useBaseline) return;

        double mean = 0;
        for (StateTransition stateTransition : stateTransitions) mean += stateTransition.tdTarget;
        mean /= stateTransitions.size();

        averageMean = averageMean == Double.NEGATIVE_INFINITY ? mean : tau * averageMean + (1 - tau) * mean;

        double std = 0;
        for (StateTransition stateTransition : stateTransitions) std += Math.pow(stateTransition.tdTarget - mean, 2);
        std = Math.sqrt(std / (stateTransitions.size() - 1));

        averageStd = averageStd == Double.NEGATIVE_INFINITY ? std : tau * averageStd + (1 - tau) * std;

        for (StateTransition stateTransition : stateTransitions) stateTransition.tdTarget = (stateTransition.tdTarget - averageMean) / averageStd;
    }

    /**
     * Returns FunctionEstimator.
     *
     * @return FunctionEstimator.
     */
    public FunctionEstimator getFunctionEstimator() {
        return functionEstimator;
    }

    /**
     * Updates FunctionEstimator.
     *
     * @param agent agent.
     */
    public void updateFunctionEstimator(Agent agent) {
        if (getFunctionEstimator().sampledSetEmpty()) return;
        getFunctionEstimator().update(getFunctionEstimator().getSampledStateTransitions());
    }

}
