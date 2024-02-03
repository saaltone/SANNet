/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import utils.configurable.DynamicParamException;

import java.util.Objects;
import java.util.TreeSet;

/**
 * Implements entropy noisy next best policy.<br>
 * Policy makes a greedy decision or next best policy according to exploration probability coming from action value entropy.<br>
 *
 */
public class EntropyNoisyNextBestPolicy extends AbstractExecutablePolicy {

    /**
     * Constructor for entropy noisy next best policy.
     *
     */
    public EntropyNoisyNextBestPolicy() {
        super(ExecutablePolicyType.ENTROPY_NOISY_NEXT_BEST);
    }

    /**
     * Constructor for entropy noisy next best policy.
     *
     * @param params parameters for Policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public EntropyNoisyNextBestPolicy(String params) throws DynamicParamException {
        super(ExecutablePolicyType.ENTROPY_NOISY_NEXT_BEST, params, null);
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
        if (stateValueSet.size() > 1 && Math.random() < getActionEntropy(stateValueSet)) stateValueSet.pollLast();
        return stateValueSet.isEmpty() ? -1 : Objects.requireNonNull(stateValueSet.pollLast()).action();
    }

}
