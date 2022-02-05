/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
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
     * Executable policy type.
     *
     */
    private final ExecutablePolicyType executablePolicyType = ExecutablePolicyType.GREEDY;

    /**
     * Constructor for greedy policy.
     *
     */
    public GreedyPolicy() {
    }

    /**
     * Constructor for greedy policy.
     *
     * @param params parameters for Policy.
     * @param paramNameTypes parameter names types
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public GreedyPolicy(String params, String paramNameTypes) throws DynamicParamException {
        super(params, paramNameTypes);
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

    /**
     * Returns executable policy type.
     *
     * @return executable policy type.
     */
    public ExecutablePolicyType getExecutablePolicyType() {
        return executablePolicyType;
    }

}
