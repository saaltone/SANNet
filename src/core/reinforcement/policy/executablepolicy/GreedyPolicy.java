/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import utils.configurable.DynamicParamException;

import java.util.Objects;
import java.util.TreeSet;

/**
 * Implements greedy policy.<br>
 *
 */
public class GreedyPolicy extends AbstractExecutablePolicy {

    /**
     * Constructor for greedy policy.
     *
     */
    public GreedyPolicy() {
        super(ExecutablePolicyType.GREEDY);
    }

    /**
     * Constructor for greedy policy.
     *
     * @param executablePolicyType executable policy type.
     */
    protected GreedyPolicy(ExecutablePolicyType executablePolicyType) {
        super(executablePolicyType);
    }

    /**
     * Constructor for greedy policy.
     *
     * @param params parameters for Policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public GreedyPolicy(String params) throws DynamicParamException {
        super(ExecutablePolicyType.GREEDY, params, null);
    }

    /**
     * Constructor for greedy policy.
     *
     * @param executablePolicyType executable policy type.
     * @param params parameters for Policy.
     * @param paramNameTypes parameter names types
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected GreedyPolicy(ExecutablePolicyType executablePolicyType, String params, String paramNameTypes) throws DynamicParamException {
        super(executablePolicyType, params, paramNameTypes);
    }

    /**
     * Increments policy.
     *
     */
    public void increment() {
    }

    /**
     * Returns action based on policy.
     *
     * @param stateValueSet priority queue containing action values in decreasing order.
     * @return chosen action.
     */
    protected int getAction(TreeSet<ActionValueTuple> stateValueSet) {
        return stateValueSet.isEmpty() ? -1 : Objects.requireNonNull(stateValueSet.pollLast()).action();
    }

}
