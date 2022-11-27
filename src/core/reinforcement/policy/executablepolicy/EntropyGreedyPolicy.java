/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import java.util.Random;
import java.util.TreeSet;

/**
 * Implements entropy greedy policy.<br>
 * Policy makes a greedy (exploit) or random (explore) decision according to exploration probability coming from action value entropy.<br>
 *
 */
public class EntropyGreedyPolicy extends GreedyPolicy {

    /**
     * Random function for entropy greedy policy.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for entropy greedy policy.
     *
     */
    public EntropyGreedyPolicy() {
        super(ExecutablePolicyType.ENTROPY_GREEDY);
    }

    /**
     * Returns action based on policy.
     *
     * @param stateValueSet priority queue containing action values in decreasing order.
     * @return chosen action.
     */
    protected int getAction(TreeSet<AbstractExecutablePolicy.ActionValueTuple> stateValueSet) {
        if (Math.random() < getActionEntropy(stateValueSet)) {
            AbstractExecutablePolicy.ActionValueTuple[] actionValueTupleArray = new AbstractExecutablePolicy.ActionValueTuple[stateValueSet.size()];
            actionValueTupleArray = stateValueSet.toArray(actionValueTupleArray);
            return actionValueTupleArray[random.nextInt(actionValueTupleArray.length)].action();
        }
        else return super.getAction(stateValueSet);
    }

}
