/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.value;

import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.function.DirectFunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.agent.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;

import java.util.TreeSet;

/**
 * Implements plain value function without function estimator.<br>
 *
 */
public class PlainValueFunction extends AbstractValueFunction {

    /**
     * Parameter name types for plain value function.
     *     - useBaseline: if true baseline is used for value function. Default value true.<br>
     *     - tau: tau value for baseline (mean and standard deviation) averaging. Default value 0.95.<br>
     *
     */
    private final static String paramNameTypes = "(useBaseline:BOOLEAN), " +
            "(tau:DOUBLE)";

    /**
     * Reference to direct function estimator.
     *
     */
    private final DirectFunctionEstimator functionEstimator;

    /**
     * If true uses baseline for target value update.
     *
     */
    private boolean useBaseline;

    /**
     * Tau value for baseline (mean and standard deviation) averaging.
     *
     */
    private double tau;

    /**
     * Average mean.
     *
     */
    private double averageMean = Double.MIN_VALUE;

    /**
     * Average standard deviation.
     *
     */
    private double averageStandardDeviation = Double.MIN_VALUE;

    /**
     * Constructor for plain value function.
     *
     * @param functionEstimator reference to direct function estimator.
     */
    public PlainValueFunction(DirectFunctionEstimator functionEstimator) {
        this(1, functionEstimator);
    }

    /**
     * Constructor for plain value function.
     *
     * @param functionEstimator reference to direct function estimator.
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public PlainValueFunction(DirectFunctionEstimator functionEstimator, String params) throws DynamicParamException {
        this(1, functionEstimator, params);
    }

    /**
     * Constructor for plain value function.
     *
     * @param numberOfActions number of actions for plain value function.
     * @param functionEstimator reference to direct function estimator.
     */
    public PlainValueFunction(int numberOfActions, DirectFunctionEstimator functionEstimator) {
        super(numberOfActions);
        this.functionEstimator = functionEstimator;
    }

    /**
     * Constructor for plain value function.
     *
     * @param numberOfActions number of actions for plain value function.
     * @param functionEstimator reference to direct function estimator.
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public PlainValueFunction(int numberOfActions, DirectFunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super(numberOfActions, params);
        this.functionEstimator = functionEstimator;
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        useBaseline = true;
        tau = 0.95;
    }

    /**
     * Returns parameters used for plain value function.
     *
     * @return parameters used for plain value function.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + PlainValueFunction.paramNameTypes;
    }

    /**
     * Sets parameters used for plain value function.<br>
     * <br>
     * Supported parameters are:<br>
     *     - useBaseline: if true uses baseline for value function. Default value true.<br>
     *     - tau: tau value for baseline (mean and standard deviation) averaging. Default value 0.95.<br>
     *
     * @param params parameters used for plain value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("useBaseline")) useBaseline = params.getValueAsBoolean("useBaseline");
        if (params.hasParam("tau")) tau = params.getValueAsDouble("tau");
    }

    /**
     * Returns reference to value function.
     *
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ValueFunction reference() throws DynamicParamException {
        return new PlainValueFunction(getNumberOfActions(), (DirectFunctionEstimator)functionEstimator.reference(), getParams());
    }

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator, boolean sharedMemory) throws DynamicParamException {
        return new PlainValueFunction(getNumberOfActions(), sharedValueFunctionEstimator ? functionEstimator : (DirectFunctionEstimator)functionEstimator.reference(), getParams());
    }

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param memory reference to memory.
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator, Memory memory) throws DynamicParamException {
        return new PlainValueFunction(getNumberOfActions(), (DirectFunctionEstimator)functionEstimator.reference(memory), getParams());
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
     * Not used.
     *
     * @param agent agent.
     */
    public void registerAgent(Agent agent) {
        functionEstimator.registerAgent(agent);
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

        int size = stateTransitions.size();

        double mean = 0;
        for (StateTransition stateTransition : stateTransitions) mean += stateTransition.tdTarget;
        mean /= size;

        averageMean = averageMean == Double.MIN_VALUE ? mean : tau * averageMean + (1 - tau) * mean;

        double std = 0;
        for (StateTransition stateTransition : stateTransitions) std += Math.pow(stateTransition.tdTarget - mean, 2);
        std = Math.sqrt(std / (size - 1));

        averageStandardDeviation = averageStandardDeviation == Double.MIN_VALUE ? std : tau * averageStandardDeviation + (1 - tau) * std;

        for (StateTransition stateTransition : stateTransitions) {
            stateTransition.tdTarget = (stateTransition.tdTarget - averageMean) / averageStandardDeviation;
        }
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
     * @return true if all registered agents are ready to update.
     */
    public boolean readyToUpdate(Agent agent) throws AgentException {
        return getFunctionEstimator().readyToUpdate(agent);
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
     * Updates function estimator.
     *
     */
    public void updateFunctionEstimator() {
        TreeSet<StateTransition> sampledStateTransitions = getSampledStateTransitions();
        if (sampledStateTransitions == null) {
            functionEstimator.abortUpdate();
            return;
        }

        getFunctionEstimator().update(sampledStateTransitions);
    }

    /**
     * Appends parameters to this value function from another value function.
     *
     * @param valueFunction value function used to update current value function.
     * @param tau tau which controls contribution of other value function.
     */
    public void append(ValueFunction valueFunction, double tau) {
    }

}
