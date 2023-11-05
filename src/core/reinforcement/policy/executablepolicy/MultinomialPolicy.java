/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;

import java.util.Objects;
import java.util.Random;
import java.util.TreeSet;

/**
 * Implements multinomial policy.<br>
 *
 */
public class MultinomialPolicy extends AbstractExecutablePolicy {

    /**
     * Parameter name types for multinomial policy.
     *     - numberOfTrials: number of trials. Default value 1.<br>
     *
     */
    private final static String paramNameTypes = "(numberOfTrials:INT)";

    /**
     * Random function for multinomial policy.
     *
     */
    private final Random random = new Random();

    /**
     * Number of trials
     *
     */
    private int numberOfTrials;

    /**
     * Constructor for multinomial policy.
     *
     * @param params parameters for multinomial policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MultinomialPolicy(String params) throws DynamicParamException {
        super(ExecutablePolicyType.MULTINOMIAL, params, MultinomialPolicy.paramNameTypes);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        numberOfTrials = 1;
    }

    /**
     * Returns parameters used for multinomial policy.
     *
     * @return parameters used for multinomial policy.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + MultinomialPolicy.paramNameTypes;
    }

    /**
     * Sets parameters used for multinomial policy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - numberOfTrials: number of trials. Default value 1.<br>
     *
     * @param params parameters used for multinomial policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("numberOfTrials")) numberOfTrials = params.getValueAsInteger("numberOfTrials");
        if (numberOfTrials < 1) throw new DynamicParamException("Number of trials cannot be less than 1.");
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
        return stateValueSet.isEmpty() ? -1 : Objects.requireNonNull(getMultinomial(1, stateValueSet).pollLast()).action();
    }

    /**
     * Returns binomial distribution.
     * Reference: <a href="https://peterchng.com/blog/2020/10/23/building-binomial-and-multinomial-samplers-in-java/">...</a>
     *
     * @param numberOfTrials number of trials.
     * @param probability probability.
     * @return number of successful trials.
     */
    private int getBinomial(int numberOfTrials, double probability) {
        if (numberOfTrials < 1 || probability < 0) return 0;
        if (probability > 1) return numberOfTrials;

        int numberOfSuccessfulTrials = 0;
        for (int trial = 0; trial < numberOfTrials; trial++) {
            if (random.nextDouble() < probability) numberOfSuccessfulTrials++;
        }
        return numberOfSuccessfulTrials;
    }

    /**
     * Returns multinomial distribution.
     * Reference: <a href="https://peterchng.com/blog/2020/10/23/building-binomial-and-multinomial-samplers-in-java/">...</a>
     *
     * @param numberOfTrials number of trials.
     * @param stateValueSet state value set.
     * @return multinomial distribution.
     */
    public TreeSet<AbstractExecutablePolicy.ActionValueTuple> getMultinomial(int numberOfTrials, TreeSet<AbstractExecutablePolicy.ActionValueTuple> stateValueSet) {

        TreeSet<AbstractExecutablePolicy.ActionValueTuple> result = new TreeSet<>(stateValueSet);
        result.clear();
        TreeSet<Integer> actions = new TreeSet<>();
        for (AbstractExecutablePolicy.ActionValueTuple actionValueTuple : stateValueSet) actions.add(actionValueTuple.action());
        double sumLeft = 1;
        int trialsLeft = numberOfTrials;
        for (Integer action : actions) {
            ActionValueTuple currentActionValueTuple = null;
            for (AbstractExecutablePolicy.ActionValueTuple actionValueTuple : stateValueSet) {
                if (actionValueTuple.action() == action) {
                    currentActionValueTuple = actionValueTuple;
                    break;
                }
            }
            if (currentActionValueTuple != null) {
                if (sumLeft >= 0 && trialsLeft >= 0) {
                    double probability = currentActionValueTuple.value();
                    double binomial = getBinomial(trialsLeft, probability / sumLeft);
                    result.add(new ActionValueTuple(currentActionValueTuple.action(), binomial));
                    sumLeft -= probability;
                    trialsLeft -= binomial;
                }
                else result.add(new ActionValueTuple(currentActionValueTuple.action(), 0));
            }
        }

        return result;
    }

}
