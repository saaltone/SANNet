/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;

import java.util.HashSet;
import java.util.Random;

/**
 * Class that defines EpsilonGreedyPolicy.<br>
 *
 */
public class EpsilonGreedyPolicy extends GreedyPolicy {

    /**
     * Parameter name types for EpsilonGreedyPolicy.
     *     - epsilonInitial: Initial epsilon value for greediness / randomness of learning. Default value 1.<br>
     *     - epsilonMin: Lowest value for epsilon. Default value 0.2.<br>
     *     - epsilonDecayRate: Decay rate of epsilon. Default value 0.999.<br>
     *     - epsilonDecayByUpdateCount: If true epsilon decays along policy update count otherwise decays by epsilon decay rate. Default value false.<br>
     *
     */
    private final static String paramNameTypes = "(epsilonInitial:DOUBLE), " +
            "(epsilonMin:DOUBLE), " +
            "(epsilonDecayRate:DOUBLE), " +
            "(epsilonDecayByUpdateCount:BOOLEAN)";

    /**
     * Executable policy type.
     *
     */
    private final ExecutablePolicyType executablePolicyType = ExecutablePolicyType.EPSILON_GREEDY;

    /**
     * Random function for EpsilonGreedyPolicy.
     *
     */
    private final Random random = new Random();

    /**
     * Current epsilon value for EpsilonGreedyPolicy defining balance between exploration and exploitation.
     *
     */
    private double epsilon;

    /**
     * Initial epsilon value.
     *
     */
    private double epsilonInitial;

    /**
     * Minimum value for epsilon.
     *
     */
    private double epsilonMin;

    /**
     * Decay rate for epsilon if number of episodes is not used for epsilon decay.
     *
     */
    private double epsilonDecayRate;

    /**
     * If true epsilon decays along update count otherwise decays by epsilon decay rate.
     *
     */
    private boolean epsilonDecayByUpdateCount;

    /**
     * Count for policy updates.
     *
     */
    private int epsilonUpdateCount = 1;

    /**
     * Constructor for EpsilonGreedyPolicy.
     *
     */
    public EpsilonGreedyPolicy() {
        super();
    }

    /**
     * Constructor for EpsilonGreedyPolicy.
     *
     * @param params parameters for EpsilonGreedyPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public EpsilonGreedyPolicy(String params) throws DynamicParamException {
        super(params, EpsilonGreedyPolicy.paramNameTypes);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        epsilonInitial = 1;
        epsilonMin = 0.2;
        epsilonDecayRate = 0.999;
        epsilonDecayByUpdateCount = false;
        epsilon = epsilonInitial;
    }

    /**
     * Returns parameters used for EpsilonGreedyPolicy.
     *
     * @return parameters used for EpsilonGreedyPolicy.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + EpsilonGreedyPolicy.paramNameTypes;
    }

    /**
     * Sets parameters used for EpsilonGreedyPolicy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - epsilonInitial: Initial epsilon value for greediness / randomness of learning. Default value 1.<br>
     *     - epsilonMin: Lowest value for epsilon. Default value 0.2.<br>
     *     - epsilonDecayRate: Decay rate of epsilon. Default value 0.999.<br>
     *     - epsilonDecayByUpdateCount: If true epsilon decays along policy update count otherwise decays by epsilon decay rate. Default value false.<br>
     *
     * @param params parameters used for EpsilonGreedyPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("epsilonInitial")) epsilonInitial = params.getValueAsDouble("epsilonInitial");
        if (params.hasParam("epsilonMin")) epsilonMin = params.getValueAsDouble("epsilonMin");
        if (params.hasParam("epsilonDecayRate")) epsilonDecayRate = params.getValueAsDouble("epsilonDecayRate");
        if (params.hasParam("epsilonDecayByUpdateCount")) epsilonDecayByUpdateCount = params.getValueAsBoolean("epsilonDecayByUpdateCount");
        epsilon = epsilonInitial;
    }

    /**
     * Increments policy.
     *
     */
    public void increment() {
        if (epsilon > epsilonMin) {
            if (epsilonDecayByUpdateCount) epsilon = epsilonInitial / (double)epsilonUpdateCount++;
            else epsilon *= epsilonDecayRate;
        }
    }

    /**
     * Takes epsilon greedy action.
     *
     * @param policyValueMatrix current state value matrix.
     * @param availableActions available actions in current state
     * @param alwaysGreedy if true greedy action is always taken.
     * @return action taken.
     */
    public int action(Matrix policyValueMatrix, HashSet<Integer> availableActions, boolean alwaysGreedy) {
        if (Math.random() < epsilon) {
            Object[] availableActionsArray = availableActions.toArray();
            return (int)availableActionsArray[random.nextInt(availableActionsArray.length)];
        }
        else return super.action(policyValueMatrix, availableActions, alwaysGreedy);
    }

    /**
     * Returns executable policy type.
     *
     * @return executable policy type.
     */
    public ExecutablePolicyType getExecutablePolicyType() {
        return executablePolicyType;
    }

}
