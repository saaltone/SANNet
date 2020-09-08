/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.Matrix;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.PriorityQueue;

/**
 * Class that defines AbstractExecutablePolicy which contains shared functions for executable policies.
 *
 */
public abstract class AbstractExecutablePolicy implements ExecutablePolicy, Serializable {

    private static final long serialVersionUID = -3999341188546094490L;

    /**
     * ActionValueTuple for policy.
     *
     */
    protected static class ActionValueTuple {

        /**
         * Action.
         *
         */
        final int action;

        /**
         * Value for action.
         *
         */
        final double value;

        /**
         * Constructor for ActionValueTuple.
         *
         * @param action action value.
         * @param value value for action.
         */
        ActionValueTuple(int action, double value) {
            this.action = action;
            this.value = value;
        }

    }

    /**
     * True if values are recorded as softmax values (e^x).
     *
     */
    protected boolean asSoftMax = false;

    /**
     * Default constructor for AbstractExecutablePolicy.
     *
     */
    AbstractExecutablePolicy() {
    }

    /**
     * Default constructor for AbstractExecutablePolicy.
     *
     * @param params parameters for AbstractExecutablePolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    AbstractExecutablePolicy(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for AbstractExecutablePolicy.
     *
     * @return parameters used for AbstractExecutablePolicy.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("asSoftMax", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for AbstractExecutablePolicy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - asSoftMax: true if action values are recorded as softmax values (e^x).<br>
     *
     * @param params parameters used for AbstractExecutablePolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("asSoftMax")) asSoftMax = params.getValueAsBoolean("asSoftMax");
    }

    /**
     * Increments policy.
     *
     */
    public abstract void increment();

    /**
     * Takes action based on policy.
     *
     * @param stateValueMatrix current state value matrix.
     * @param availableActions available actions in current state
     * @param stateValueOffset state value offset
     * @return action taken.
     */
    public int action(Matrix stateValueMatrix, HashSet<Integer> availableActions, int stateValueOffset) {
        PriorityQueue<ActionValueTuple> stateValuePriorityQueue = new PriorityQueue<>((o1, o2) -> Double.compare(o2.value, o1.value));
        double cumulativeValue = 0;
        for (Integer action : availableActions) {
            double actionValue = !asSoftMax ? stateValueMatrix.getValue(action + stateValueOffset, 0) : Math.exp(stateValueMatrix.getValue(action + stateValueOffset, 0));
            cumulativeValue += actionValue;
            stateValuePriorityQueue.add(new ActionValueTuple(action, actionValue));
        }
        return getAction(stateValuePriorityQueue, cumulativeValue);
    }

    /**
     * Returns action based on policy.
     *
     * @param stateValuePriorityQueue priority queue containing action values in decreasing order.
     * @param cumulativeValue cumulative value of actions.
     * @return chosen action.
     */
    protected abstract int getAction(PriorityQueue<ActionValueTuple> stateValuePriorityQueue, double cumulativeValue);

}
