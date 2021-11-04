/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import core.reinforcement.memory.StateTransition;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Class that defines AbstractExecutablePolicy which contains shared functions for executable policies.<br>
 *
 */
public abstract class AbstractExecutablePolicy implements ExecutablePolicy, Serializable {

    @Serial
    private static final long serialVersionUID = -3999341188546094490L;

    /**
     * Parameter name types for AbstractExecutablePolicy.
     *     - asSoftMax: true if action values are recorded as softmax values (e^x).<br>
     *
     */
    private final static String paramNameTypes = "(asSoftMax:BOOLEAN)";

    /**
     * Record that defines ActionValueTuple for policy.
     *
     * @param action action value.
     * @param value value for action.
     */
    protected record ActionValueTuple(int action, double value) {
    }

    /**
     * True if values are recorded as softmax values (e^x).
     *
     */
    protected boolean asSoftMax;

    /**
     * Default constructor for AbstractExecutablePolicy.
     *
     */
    AbstractExecutablePolicy() {
        initializeDefaultParams();
    }

    /**
     * Default constructor for AbstractExecutablePolicy.
     *
     * @param params parameters for AbstractExecutablePolicy.
     * @param paramNameTypes parameter names types
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    AbstractExecutablePolicy(String params, String paramNameTypes) throws DynamicParamException {
        this();
        if (params != null) setParams(new DynamicParam(params, AbstractExecutablePolicy.paramNameTypes + (paramNameTypes != null ? ", " + paramNameTypes : "")));
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        asSoftMax = false;
    }

    /**
     * Returns parameters used for AbstractExecutablePolicy.
     *
     * @return parameters used for AbstractExecutablePolicy.
     */
    public String getParamDefs() {
        return AbstractExecutablePolicy.paramNameTypes;
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
     * @param policyValueMatrix current state value matrix.
     * @param availableActions available actions in current state
     * @param action action.
     */
    public void action(Matrix policyValueMatrix, HashSet<Integer> availableActions, int action) {
    }

    /**
     * Takes action based on policy.
     *
     * @param policyValueMatrix current state value matrix.
     * @param availableActions available actions in current state
     * @param alwaysGreedy if true greedy action is always taken.
     * @return action taken.
     */
    public int action(Matrix policyValueMatrix, HashSet<Integer> availableActions, boolean alwaysGreedy) {
        TreeSet<ActionValueTuple> stateValueSet = new TreeSet<>(Comparator.comparingDouble(o -> o.value));
        for (Integer action : availableActions) {
            stateValueSet.add(new ActionValueTuple(action, !asSoftMax ? policyValueMatrix.getValue(action, 0) : Math.exp(policyValueMatrix.getValue(action, 0))));
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
