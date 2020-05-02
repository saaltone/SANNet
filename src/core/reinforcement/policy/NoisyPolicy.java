/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement.policy;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.Matrix;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

/**
 * Class that defines NoisyPolicy.
 *
 */
public class NoisyPolicy implements Policy {

    /**
     * Random function for NoisyPolicy.
     *
     */
    private final Random random = new Random();

    /**
     * Variance for exploration noise of NoisyPolicy.
     *
     */
    private double explorationNoise = 0.2;

    /**
     * Minimum variance for exploration noise of NoisyPolicy.
     *
     */
    private double minExplorationNoise = 0.1;

    /**
     * Decay for exploration noise of NoisyPolicy.
     *
     */
    private double explorationNoiseDecay = 0.9999;

    /**
     * Constructor for NoisyPolicy.
     *
     */
    public NoisyPolicy() {
    }

    /**
     * Constructor for NoisyPolicy.
     *
     * @param params parameters for NoisyPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public NoisyPolicy(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for NoisyPolicy.
     *
     * @return parameters used for NoisyPolicy.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("explorationNoise", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("minExplorationNoise", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("explorationNoiseDecay", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for NoisyPolicy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - explorationNoise: Variance for Gaussian exploration noise for noisy policy. Default value 0.2.<br>
     *     - minExplorationNoise: Minimum variance for Gaussian exploration noise for noisy policy. Default value 0.05.<br>
     *     - explorationNoiseDecay: Decay factor for exploration noise. Default value 0.9999.<br>
     *
     * @param params parameters used for NoisyPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("explorationNoise")) explorationNoise = params.getValueAsDouble("explorationNoise");
        if (params.hasParam("minExplorationNoise")) minExplorationNoise = params.getValueAsDouble("minExplorationNoise");
        if (params.hasParam("explorationNoiseDecay")) explorationNoiseDecay = params.getValueAsDouble("explorationNoiseDecay");
    }

    /**
     * Sets current episode count.
     *
     * @param episodeCount current episode count.
     */
    public void setEpisode(int episodeCount) {}

    /**
     * Takes noisy action.
     *
     * @param stateMatrix current state matrix.
     * @param availableActions available actions in current state
     * @return action taken.
     */
    public int action(Matrix stateMatrix, HashSet<Integer> availableActions) {
        int action = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (Integer row : availableActions) {
            double actionValue = stateMatrix.getValue(row, 0) + random.nextGaussian() * Math.sqrt(explorationNoise);
            if (maxValue < actionValue || maxValue == Double.NEGATIVE_INFINITY) {
                maxValue =  actionValue;
                action = row;
            }
        }
        if (explorationNoise >= minExplorationNoise) explorationNoise *= explorationNoiseDecay;
        return action;
    }

}
