/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

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
     * If true uses averaged entropy for exploration / exploitation balance.
     *
     */
    private boolean useAveragingEntropy;

    /**
     * Average entropy.
     *
     */
    private transient double averageEntropy = Double.MIN_VALUE;

    /**
     * Averaging factor for average entropy.
     *
     */
    private double tau;

    /**
     * Minimum value for greedy policy.
     *
     */
    private double minThreshold;

    /**
     * Constructor for entropy greedy policy.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public EntropyGreedyPolicy() throws MatrixException {
        super(ExecutablePolicyType.ENTROPY_GREEDY);
    }

    /**
     * Constructor for entropy greedy policy.
     *
     * @param params parameters for entropy greedy policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public EntropyGreedyPolicy(String params) throws DynamicParamException, MatrixException {
        super(ExecutablePolicyType.ENTROPY_GREEDY, params, EntropyGreedyPolicy.paramNameTypes);
    }

    /**
     * Initializes default params.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void initializeDefaultParams() throws MatrixException {
        super.initializeDefaultParams();
        entropyFactor = 1;
        useAveragingEntropy = true;
        tau = 0.99;
        minThreshold = 0.05;
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
        double usedEntropy;
        if (useAveragingEntropy) {
            averageEntropy = averageEntropy == Double.MIN_VALUE ? entropy : tau * averageEntropy + (1 - tau) * entropy;
            usedEntropy = averageEntropy;
        }
        else usedEntropy = entropy;
        boolean randomChoice = Math.random() < Math.max(minThreshold, usedEntropy * entropyFactor);
        if (randomChoice) {
            AbstractExecutablePolicy.ActionValueTuple[] actionValueTupleArray = new AbstractExecutablePolicy.ActionValueTuple[stateValueSet.size()];
            actionValueTupleArray = stateValueSet.toArray(actionValueTupleArray);
            return actionValueTupleArray[random.nextInt(actionValueTupleArray.length)].action();
        }
        else return super.getAction(stateValueSet);
    }

}
