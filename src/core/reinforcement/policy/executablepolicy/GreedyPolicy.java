/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import utils.DynamicParamException;

import java.util.TreeSet;

/**
 * Class that defines GreedyPolicy.
 *
 */
public class GreedyPolicy extends AbstractExecutablePolicy {

    /**
     * Constructor for GreedyPolicy.
     *
     */
    public GreedyPolicy() {
    }

    /**
     * Constructor for GreedyPolicy.
     *
     * @param params parameters for Policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public GreedyPolicy(String params) throws DynamicParamException {
        super(params);
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
        return stateValueSet.isEmpty() ? -1 : stateValueSet.pollLast().action;
    }

}
