/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

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
     * If true epsilon decays along action count otherwise decays by epsilon decay rate.
     *
     */
    private boolean epsilonDecayByUpdate = false;

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
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.put("epsilonInitial", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonMin", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonDecayRate", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonDecayByUpdate", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for EpsilonGreedyPolicy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - epsilonInitial: Initial epsilon value for greediness / randomness of learning. Default value 1.<br>
     *     - epsilonMin: Lowest value for epsilon. Default value 0.2.<br>
     *     - epsilonDecay: Decay rate of epsilon. Default value 0.999.<br>
     *     - epsilonDecayByEpisode: If true epsilon decays along policy update count otherwise decays by epsilon decay rate. Default value true.<br>
     *
     * @param params parameters used for EpsilonGreedyPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("epsilonInitial")) epsilonInitial = params.getValueAsDouble("epsilonInitial");
        if (params.hasParam("epsilonMin")) epsilonMin = params.getValueAsDouble("epsilonMin");
        if (params.hasParam("epsilonDecayRate")) epsilonDecayRate = params.getValueAsDouble("epsilonDecayRate");
        if (params.hasParam("epsilonDecayByUpdate")) epsilonDecayByUpdate = params.getValueAsBoolean("epsilonDecayByUpdate");
        epsilon = epsilonInitial;
    }

    /**
     * Increments policy.
     *
     */
    public void increment() {
        if (epsilon > epsilonMin) {
            if (epsilonDecayByUpdate) epsilon = epsilonInitial / (double)epsilonUpdateCount++;
            else epsilon *= epsilonDecayRate;
        }
    }

    /**
     * Takes epsilon greedy action.
     *
     * @param stateValueMatrix current state value matrix.
     * @param availableActions available actions in current state
     * @param stateValueOffset state value offset
     * @return action taken.
     */
    public int action(Matrix stateValueMatrix, HashSet<Integer> availableActions, int stateValueOffset) {
        if (Math.random() < epsilon) {
            Object[] availableActionsArray = availableActions.toArray();
            return (int)availableActionsArray[random.nextInt(availableActionsArray.length)];
        }
        else return super.action(stateValueMatrix, availableActions, stateValueOffset);
    }

}