/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import core.reinforcement.agent.AgentException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.util.Random;
import java.util.TreeSet;

/**
 * Implements epsilon greedy policy. Greediness is defined by entropy of action values.<br>
 *
 */
public class EpsilonGreedyPolicy extends GreedyPolicy {

    /**
     * Parameter name types for epsilon greedy policy.
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
     * Random function for epsilon greedy policy.
     *
     */
    private final Random random = new Random();

    /**
     * Current epsilon value for epsilon greedy policy defining balance between exploration and exploitation.
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
     * Decay-rate for epsilon if number of episodes is not used for epsilon decay.
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
     * Constructor for epsilon greedy policy.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public EpsilonGreedyPolicy() throws MatrixException {
        super(ExecutablePolicyType.EPSILON_GREEDY);
    }

    /**
     * Constructor for epsilon greedy policy.
     *
     * @param params parameters for epsilon greedy policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public EpsilonGreedyPolicy(String params) throws DynamicParamException, MatrixException {
        super(ExecutablePolicyType.EPSILON_GREEDY, params, EpsilonGreedyPolicy.paramNameTypes);
    }

    /**
     * Initializes default params.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void initializeDefaultParams() throws MatrixException {
        super.initializeDefaultParams();
        epsilonInitial = 1;
        epsilonMin = 0.2;
        epsilonDecayRate = 0.999;
        epsilonDecayByUpdateCount = false;
        epsilon = epsilonInitial;
    }

    /**
     * Returns parameters used for epsilon greedy policy.
     *
     * @return parameters used for epsilon greedy policy.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + EpsilonGreedyPolicy.paramNameTypes;
    }

    /**
     * Sets parameters used for epsilon greedy policy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - epsilonInitial: Initial epsilon value for greediness / randomness of learning. Default value 1.<br>
     *     - epsilonMin: Lowest value for epsilon. Default value 0.2.<br>
     *     - epsilonDecayRate: Decay rate of epsilon. Default value 0.999.<br>
     *     - epsilonDecayByUpdateCount: If true epsilon decays along policy update count otherwise decays by epsilon decay rate. Default value false.<br>
     *
     * @param params parameters used for epsilon greedy policy.
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
            epsilon = epsilonDecayByUpdateCount ? epsilonInitial / (double)epsilonUpdateCount++ : epsilon * epsilonDecayRate;
        }
    }

    /**
     * Returns action based on policy.
     *
     * @param stateValueSet priority queue containing action values in decreasing order.
     * @return chosen action.
     * @throws AgentException throws exception if policy fails to choose valid action.
     */
    protected int getAction(TreeSet<ActionValueTuple> stateValueSet) throws AgentException {
        if (stateValueSet.isEmpty()) throw new AgentException("Noisy next best policy failed to choose valid action.");
        else {
            if (Math.random() < epsilon) {
                ActionValueTuple[] actionValueTupleArray = new ActionValueTuple[stateValueSet.size()];
                actionValueTupleArray = stateValueSet.toArray(actionValueTupleArray);
                return actionValueTupleArray[random.nextInt(actionValueTupleArray.length)].action();
            }
            else return super.getAction(stateValueSet);
        }
    }

}
