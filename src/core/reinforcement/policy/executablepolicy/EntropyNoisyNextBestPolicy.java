/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import core.reinforcement.agent.AgentException;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

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
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public EntropyNoisyNextBestPolicy() throws MatrixException {
        super(ExecutablePolicyType.ENTROPY_NOISY_NEXT_BEST);
    }

    /**
     * Constructor for entropy noisy next best policy.
     *
     * @param params parameters for Policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public EntropyNoisyNextBestPolicy(String params) throws DynamicParamException, MatrixException {
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
     * @throws AgentException throws exception if policy fails to choose valid action.
     */
    protected int getAction(TreeSet<ActionValueTuple> stateValueSet) throws AgentException {
        if (stateValueSet.size() > 1 && Math.random() < getActionEntropy(stateValueSet)) stateValueSet.pollLast();
        if (stateValueSet.isEmpty()) throw new AgentException("Noisy next best policy failed to choose valid action.");
        else {
            ActionValueTuple actionValueTuple = stateValueSet.pollLast();
            if (actionValueTuple == null) throw new AgentException("Noisy next best policy failed to choose valid action.");
            else return actionValueTuple.action();
        }
    }

}
