/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.value;

import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.function.DirectFunctionEstimator;
import core.reinforcement.agent.State;
import core.reinforcement.function.FunctionEstimator;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.TreeSet;

/**
 * Implements plain value function without function estimator.<br>
 *
 */
public class PlainValueFunction extends AbstractActionValueFunction {

    /**
     * Parameter name types for plain value function.
     *     - useBaseline: if true baseline is used for value function. Default value true.<br>
     *     - tau: tau value for baseline (mean and standard deviation) averaging. Default value 0.99.<br>
     *
     */
    private final static String paramNameTypes = "(useBaseline:BOOLEAN), " +
            "(tau:DOUBLE)";

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
    private double averageMean = 0;

    /**
     * Average standard deviation.
     *
     */
    private double averageStandardDeviation = 0;

    /**
     * Constructor for plain value function.
     *
     * @param functionEstimator reference to direct function estimator.
     * @param params parameters for value function.
     */
    public PlainValueFunction(DirectFunctionEstimator functionEstimator, String params) {
        super(functionEstimator, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        useBaseline = true;
        tau = 0.99;
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
     *     - useBaseline: if true, uses baseline for value function. Default value true.<br>
     *     - tau: tau value for baseline (mean and standard deviation) averaging. Default value 0.99.<br>
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
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @return reference to value function.
     * @throws IOException            throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws MatrixException        throws exception if neural network has less output than actions.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new PlainValueFunction((DirectFunctionEstimator)functionEstimator.reference(), getParams());
    }

    /**
     * Not used.
     *
     */
    public void start(Agent agent) {
        getFunctionEstimator().registerAgent(agent);
    }

    /**
     * Not used.
     *
     */
    public void stop() {}

    /**
     * Updates state value.
     *
     * @param state state.
     */
    protected double getStateValue(State state) {
        return state.value;
    }

    /**
     * Returns target value based on next state.
     *
     * @param nextState next state.
     * @return target value based on next state
     */
    public double getTargetValue(State nextState) {
        return useBaseline && averageMean > 0 && averageStandardDeviation > 0? (nextState.tdTarget - averageMean) / averageStandardDeviation : nextState.tdTarget;
    }

    /**
     * Returns target action based on next state.
     *
     * @param nextState next state.
     * @return target action based on next state
     */
    protected int getTargetAction(State nextState) {
        return nextState.action;
    }

    /**
     * Updates baseline values for states.
     *
     * @param states states.
     */
    protected void updateBaseline(TreeSet<State> states) {
        if (!useBaseline) return;

        int size = states.size();
        if (size < 2) return;

        double mean = 0;
        for (State state : states) mean += state.value;
        mean /= size;
        averageMean = averageMean == 0 ? mean : tau * averageMean + (1 - tau) * mean;

        double standardDeviation = 0;
        for (State state : states) standardDeviation += Math.pow(state.value - mean, 2);
        standardDeviation = Math.sqrt(standardDeviation / (size - 1));
        averageStandardDeviation = averageStandardDeviation == 0 ? standardDeviation : tau * averageStandardDeviation + (1 - tau) * standardDeviation;

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
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @return true if all registered agents are ready to update.
     */
    public boolean readyToUpdate(Agent agent) throws AgentException {
        return getFunctionEstimator().readyToUpdate(agent);
    }

    /**
     * Updates function estimator.
     *
     * @param sampledStates sampled states.
     */
    public void updateFunctionEstimator(TreeSet<State> sampledStates) {
    }

}
