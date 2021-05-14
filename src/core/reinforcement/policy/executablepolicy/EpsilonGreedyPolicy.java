/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.Matrix;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

/**
 * Class that defines EpsilonGreedyPolicy.<br>
 *
 */
public class EpsilonGreedyPolicy extends GreedyPolicy {

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
    private double epsilonInitial = 1;

    /**
     * Minimum value for epsilon.
     *
     */
    private double epsilonMin = 0.2;

    /**
     * Decay rate for epsilon if number of episodes is not used for epsilon decay.
     *
     */
    private double epsilonDecayRate = 0.999;

    /**
     * If true epsilon decays along update count otherwise decays by epsilon decay rate.
     *
     */
    private boolean epsilonDecayByUpdateCount = false;

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
        epsilon = epsilonInitial;
    }

    /**
     * Constructor for EpsilonGreedyPolicy.
     *
     * @param params parameters for EpsilonGreedyPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public EpsilonGreedyPolicy(String params) throws DynamicParamException {
        super(params);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for EpsilonGreedyPolicy.
     *
     * @return parameters used for EpsilonGreedyPolicy.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.put("epsilonInitial", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonMin", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonDecayRate", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonDecayByUpdateCount", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
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
     * @param stateValueMatrix current state value matrix.
     * @param availableActions available actions in current state
     * @param stateValueOffset state value offset
     * @param alwaysGreedy if true greedy action is always taken.
     * @return action taken.
     */
    public int action(Matrix stateValueMatrix, HashSet<Integer> availableActions, int stateValueOffset, boolean alwaysGreedy) {
        if (Math.random() < epsilon) {
            Object[] availableActionsArray = availableActions.toArray();
            return (int)availableActionsArray[random.nextInt(availableActionsArray.length)];
        }
        else return super.action(stateValueMatrix, availableActions, stateValueOffset, alwaysGreedy);
    }

}
