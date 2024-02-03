/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

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
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public GreedyPolicy() throws MatrixException {
        super(ExecutablePolicyType.GREEDY);
    }

    /**
     * Constructor for greedy policy.
     *
     * @param executablePolicyType executable policy type.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected GreedyPolicy(ExecutablePolicyType executablePolicyType) throws MatrixException {
        super(executablePolicyType);
    }

    /**
     * Constructor for greedy policy.
     *
     * @param params parameters for Policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public GreedyPolicy(String params) throws DynamicParamException, MatrixException {
        super(ExecutablePolicyType.GREEDY, params, null);
    }

    /**
     * Constructor for greedy policy.
     *
     * @param executablePolicyType executable policy type.
     * @param params parameters for Policy.
     * @param paramNameTypes parameter names types
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected GreedyPolicy(ExecutablePolicyType executablePolicyType, String params, String paramNameTypes) throws DynamicParamException, MatrixException {
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
