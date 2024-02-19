/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import core.reinforcement.agent.AgentException;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.util.Random;
import java.util.TreeSet;

/**
 * Implements multinomial policy.<br>
 *
 */
public class MultinomialPolicy extends AbstractExecutablePolicy {

    /**
     * Random function for multinomial policy.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for multinomial policy.
     *
     * @param params parameters for multinomial policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MultinomialPolicy(String params) throws DynamicParamException, MatrixException {
        super(ExecutablePolicyType.MULTINOMIAL, params, null);
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
    protected int getAction(TreeSet<AbstractExecutablePolicy.ActionValueTuple> stateValueSet) throws AgentException {
        if (stateValueSet.isEmpty()) throw new AgentException("Noisy next best policy failed to choose valid action.");
        else return getRandomChoice(stateValueSet);
    }

    /**
     * Returns weighted random choice.
     *
     * @param stateValueSet state value set.
     * @return chosen action.
     * @throws AgentException throws exception if policy fails to choose valid action.
     */
    public int getRandomChoice(TreeSet<AbstractExecutablePolicy.ActionValueTuple> stateValueSet) throws AgentException {
        double valueSum = 0;
        for (ActionValueTuple actionValueTuple : stateValueSet) valueSum += actionValueTuple.value();

        double threshold = valueSum * random.nextDouble();

        valueSum = 0;
        for (ActionValueTuple actionValueTuple : stateValueSet) {
            valueSum += actionValueTuple.value();
            if (valueSum >= threshold) return actionValueTuple.action();
        }

        throw new AgentException("Multinomial policy failed to choose valid action.");
    }

}
