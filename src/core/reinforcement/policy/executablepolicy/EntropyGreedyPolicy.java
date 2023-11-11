/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;

import java.util.Random;
import java.util.TreeSet;

/**
 * Implements entropy greedy policy.<br>
 * Policy makes a greedy (exploit) or random (explore) decision according to exploration probability coming from action value entropy.<br>
 *
 */
public class EntropyGreedyPolicy extends GreedyPolicy {

    /**
     * Parameter name types for entropy greedy policy.
     *     - entropyFactor: factor for controlling randomness of policy. Default value 0.3.<br>
     *
     */
    private final static String paramNameTypes = "(entropyFactor:DOUBLE)";

    /**
     * Random function for entropy greedy policy.
     *
     */
    private final Random random = new Random();

    /**
     * Factor for controlling randomness of policy.
     *
     */
    private double entropyFactor;

    /**
     * Constructor for entropy greedy policy.
     *
     */
    public EntropyGreedyPolicy() {
        super(ExecutablePolicyType.ENTROPY_GREEDY);
    }

    /**
     * Constructor for entropy greedy policy.
     *
     * @param params parameters for entropy greedy policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public EntropyGreedyPolicy(String params) throws DynamicParamException {
        super(ExecutablePolicyType.ENTROPY_GREEDY, params, EntropyGreedyPolicy.paramNameTypes);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        entropyFactor = 0.3;
    }

    /**
     * Returns parameters used for entropy greedy policy.
     *
     * @return parameters used for entropy greedy policy.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + EntropyGreedyPolicy.paramNameTypes;
    }

    /**
     * Sets parameters used for epsilon greedy policy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - entropyFactor: factor for controlling randomness of policy. Default value 0.3.<br>
     *
     * @param params parameters used for epsilon greedy policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("entropyFactor")) entropyFactor = params.getValueAsDouble("entropyFactor");
    }

    /**
     * Returns action based on policy.
     *
     * @param stateValueSet priority queue containing action values in decreasing order.
     * @return chosen action.
     */
    protected int getAction(TreeSet<AbstractExecutablePolicy.ActionValueTuple> stateValueSet) {
        double entropy = getActionEntropy(stateValueSet);
        boolean randomChoice = Math.random() < entropy * entropyFactor;
        if (randomChoice) {
            AbstractExecutablePolicy.ActionValueTuple[] actionValueTupleArray = new AbstractExecutablePolicy.ActionValueTuple[stateValueSet.size()];
            actionValueTupleArray = stateValueSet.toArray(actionValueTupleArray);
            return actionValueTupleArray[random.nextInt(actionValueTupleArray.length)].action();
        }
        else return super.getAction(stateValueSet);
    }

}
