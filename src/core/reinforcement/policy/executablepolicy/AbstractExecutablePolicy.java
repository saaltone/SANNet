/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.State;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Implements abstract executable policy which contains shared functions for executable policies.<br>
 *
 */
public abstract class AbstractExecutablePolicy implements ExecutablePolicy, Serializable {

    @Serial
    private static final long serialVersionUID = -3999341188546094490L;

    /**
     * Parameter name types for abstract executable policy.
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
     * Executable policy type.
     *
     */
    private final ExecutablePolicyType executablePolicyType;

    /**
     * True if values are recorded as softmax values (e^x).
     *
     */
    protected boolean asSoftMax;

    /**
     * If true policy is learning otherwise not.
     *
     */
    private boolean isLearning = true;

    /**
     * Default constructor for abstract executable policy.
     *
     * @param executablePolicyType executable policy type.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    AbstractExecutablePolicy(ExecutablePolicyType executablePolicyType) throws MatrixException {
        this.executablePolicyType = executablePolicyType;
        initializeDefaultParams();
    }

    /**
     * Default constructor for abstract executable policy.
     *
     * @param executablePolicyType executable policy type.
     * @param params parameters for abstract executable policy.
     * @param paramNameTypes parameter names types
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    AbstractExecutablePolicy(ExecutablePolicyType executablePolicyType, String params, String paramNameTypes) throws DynamicParamException, MatrixException {
        this(executablePolicyType);
        if (params != null) setParams(new DynamicParam(params, AbstractExecutablePolicy.paramNameTypes + (paramNameTypes != null ? ", " + paramNameTypes : "")));
    }

    /**
     * Initializes default params.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void initializeDefaultParams() throws MatrixException {
        asSoftMax = false;
    }

    /**
     * Returns parameters used for abstract executable policy.
     *
     * @return parameters used for abstract executable policy.
     */
    public String getParamDefs() {
        return AbstractExecutablePolicy.paramNameTypes;
    }

    /**
     * Sets parameters used for abstract executable policy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - asSoftMax: true if action values are recorded as softmax values (e^x).<br>
     *
     * @param params parameters used for abstract executable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("asSoftMax")) asSoftMax = params.getValueAsBoolean("asSoftMax");
    }

    /**
     * Sets flag if agent is in learning mode.
     *
     * @param isLearning if true agent is in learning mode.
     */
    public void setLearning(boolean isLearning) {
        this.isLearning = isLearning;
    }

    /**
     * Returns if policy is learning or not.
     *
     * @return returns true is policy is learning otherwise not.
     */
    protected boolean isLearning() {
        return isLearning;
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
     * @param availableActions  available actions in current state
     * @return action taken.
     * @throws AgentException throws exception if policy fails to choose valid action.
     */
    public int action(Matrix policyValueMatrix, HashSet<Integer> availableActions) throws AgentException {
        TreeSet<ActionValueTuple> stateValueSet = new TreeSet<>(Comparator.comparingDouble(o -> o.value));
        for (Integer action : availableActions) {
            stateValueSet.add(new ActionValueTuple(action, !asSoftMax ? policyValueMatrix.getValue(action, 0, 0) : Math.exp(policyValueMatrix.getValue(action, 0, 0))));
        }
        return stateValueSet.isEmpty() ? -1 : !isLearning() ? Objects.requireNonNull(stateValueSet.pollLast()).action : getAction(stateValueSet);
    }

    /**
     * Returns action entropy
     *
     * @param stateValueSet action value set.
     * @return action entropy.
     */
    protected double getActionEntropy(TreeSet<ActionValueTuple> stateValueSet) {
        double entropy = 0;
        double base = stateValueSet.size() > 1 ? Math.log(stateValueSet.size()) : 1;
        for (ActionValueTuple actionValueTuple : stateValueSet) {
            entropy += Math.log(actionValueTuple.value) * actionValueTuple.value / base;
        }
        return -entropy;
    }

    /**
     * Adds state for action execution.
     *
     * @param state state.
     */
    public void add(State state) {
    }

    /**
     * Ends episode.
     *
     */
    public void endEpisode() {
    }

    /**
     * Returns action based on policy.
     *
     * @param stateValueSet priority queue containing action values in decreasing order.
     * @return chosen action.
     * @throws AgentException throws exception if policy fails to choose valid action.
     */
    protected abstract int getAction(TreeSet<ActionValueTuple> stateValueSet) throws AgentException;

    /**
     * Returns executable policy type.
     *
     * @return executable policy type.
     */
    public ExecutablePolicyType getExecutablePolicyType() {
        return executablePolicyType;
    }

}
