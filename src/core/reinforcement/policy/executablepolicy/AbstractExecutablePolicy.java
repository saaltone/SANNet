/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import core.reinforcement.memory.StateTransition;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.Matrix;

import java.io.Serializable;
import java.util.*;

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
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
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
     * Resets policy.
     *
     * @param forceReset force reset.
     */
    public void reset(boolean forceReset) {
    }

    /**
     * Takes action decided by external agent.
     *
     * @param stateValueMatrix current state value matrix.
     * @param availableActions available actions in current state
     * @param stateValueOffset state value offset
     * @param action action.
     */
    public void action(Matrix stateValueMatrix, HashSet<Integer> availableActions, int stateValueOffset, int action) {
    }

    /**
     * Takes action based on policy.
     *
     * @param stateValueMatrix current state value matrix.
     * @param availableActions available actions in current state
     * @param stateValueOffset state value offset
     * @param alwaysGreedy if true greedy action is always taken.
     * @return action taken.
     */
    public int action(Matrix stateValueMatrix, HashSet<Integer> availableActions, int stateValueOffset, boolean alwaysGreedy) {
        TreeSet<ActionValueTuple> stateValueSet = new TreeSet<>(Comparator.comparingDouble(o -> o.value));
        for (Integer action : availableActions) {
            stateValueSet.add(new ActionValueTuple(action, !asSoftMax ? stateValueMatrix.getValue(action + stateValueOffset, 0) : Math.exp(stateValueMatrix.getValue(action + stateValueOffset, 0))));
        }
        return stateValueSet.isEmpty() ? -1 : alwaysGreedy ? Objects.requireNonNull(stateValueSet.pollLast()).action : getAction(stateValueSet);
    }

    /**
     * Records state transition for action execution.
     *
     * @param stateTransition state transition.
     */
    public void record(StateTransition stateTransition) {
    }

    /**
     * Finishes episode.
     *
     * @param update if true update executed.
     */
    public void finish(boolean update) {
    }

    /**
     * Returns action based on policy.
     *
     * @param stateValueSet priority queue containing action values in decreasing order.
     * @return chosen action.
     */
    protected abstract int getAction(TreeSet<ActionValueTuple> stateValueSet);

}
