/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.Matrix;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

/**
 * Class that defines EpsilonGreedyPolicy.
 *
 */
public class EpsilonGreedyPolicy extends GreedyPolicy {

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
    private double epsilonInitial = 1;

    /**
     * Minimum value for epsilon.
     *
     */
    private double epsilonMin = 0.1;

    /**
     * Decay rate for epsilon if number of episodes is not used for epsilon decay.
     *
     */
    private double epsilonDecayRate = 0.999;

    /**
     * If true epsilon decays along episode count otherwise decays by epsilon decay rate.
     *
     */
    private boolean epsilonDecayByEpisode = false;

    /**
     * Constructor for epsilon greedy policy
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
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for EpsilonGreedyPolicy.
     *
     * @return parameters used for EpsilonGreedyPolicy.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("epsilonInitial", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonMin", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonDecayRate", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonDecayByEpisode", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for EpsilonGreedyPolicy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - epsilonInitial: Initial epsilon value for greediness / randomness of learning. Default value 1.<br>
     *     - epsilonMin: Lowest value for epsilon. Default value 0.<br>
     *     - epsilonDecay: Decay rate of epsilon. Default value 0.99.<br>
     *     - epsilonDecayByEpisode: If true epsilon decays along episode count otherwise decays by epsilon decay rate. Default value true.<br>
     *
     * @param params parameters used for EpsilonGreedyPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("epsilonInitial")) epsilonInitial = params.getValueAsDouble("epsilonInitial");
        if (params.hasParam("epsilonMin")) epsilonMin = params.getValueAsDouble("epsilonMin");
        if (params.hasParam("epsilonDecayRate")) epsilonDecayRate = params.getValueAsDouble("epsilonDecayRate");
        if (params.hasParam("epsilonDecayByEpisode")) epsilonDecayByEpisode = params.getValueAsBoolean("epsilonDecayByEpisode");
        epsilon = epsilonInitial;
    }

    /**
     * Sets current episode count.
     *
     * @param episodeCount current episode count.
     */
    public void setEpisode(int episodeCount) {
        if (epsilon > epsilonMin) {
            if (epsilonDecayByEpisode) epsilon = epsilonInitial / (double)episodeCount;
            else epsilon *= epsilonDecayRate;
        }
    }

    /**
     * Takes epsilon greedy action.
     *
     * @param stateMatrix current state matrix.
     * @param availableActions available actions in current state
     * @return action taken.
     */
    public int action(Matrix stateMatrix, HashSet<Integer> availableActions) {
        if (Math.random() < epsilon) {
            Object[] availableActionsArray = availableActions.toArray();
            return (int)availableActionsArray[random.nextInt(availableActionsArray.length)];
        }
        else return super.action(stateMatrix, availableActions);
    }

}
