/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

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
     */
    protected int getAction(TreeSet<AbstractExecutablePolicy.ActionValueTuple> stateValueSet) {
        return stateValueSet.isEmpty() ? -1 : getRandomChoice(stateValueSet);
    }

    /**
     * Returns weighted random choice.
     *
     * @param stateValueSet state value set.
     * @return chosen action.
     */
    public int getRandomChoice(TreeSet<AbstractExecutablePolicy.ActionValueTuple> stateValueSet) {
        double valueSum = 0;
        for (ActionValueTuple actionValueTuple : stateValueSet) valueSum += actionValueTuple.value();

        double threshold = valueSum * random.nextDouble();

        valueSum = 0;
        for (ActionValueTuple actionValueTuple : stateValueSet) {
            valueSum += actionValueTuple.value();
            if (valueSum >= threshold) return actionValueTuple.action();
        }

        return -1;
    }

}
