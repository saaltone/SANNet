/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import utils.DynamicParam;
import utils.DynamicParamException;

import java.util.HashMap;
import java.util.Objects;
import java.util.Random;
import java.util.TreeSet;

/**
 * Class that defines NoisyNextBestPolicy.<br>
 * Policy make a greedy decision (chooses best policy) or next best policy according to exploration probability.<br>
 *
 */
public class NoisyNextBestPolicy extends AbstractExecutablePolicy {

    /**
     * Random function for NoisyNextBestPolicy.
     *
     */
    private final Random random = new Random();

    /**
     * Exploration noise for NoisyNextBestPolicy.
     *
     */
    private double explorationNoise;

    /**
     * Exploration noise for NoisyNextBestPolicy.
     *
     */
    private double initialExplorationNoise = 1;

    /**
     * Minimum exploration noise for NoisyNextBestPolicy.
     *
     */
    private double minExplorationNoise = 0.2;

    /**
     * Decay for exploration noise for NoisyNextBestPolicy.
     *
     */
    private double explorationNoiseDecay = 0.999;

    /**
     * Constructor for NoisyNextBestPolicy.
     *
     */
    public NoisyNextBestPolicy() {
        explorationNoise = initialExplorationNoise;
    }

    /**
     * Constructor for NoisyNextBestPolicy.
     *
     * @param params parameters for NoisyNextBestPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public NoisyNextBestPolicy(String params) throws DynamicParamException {
        super(params);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for NoisyNextBestPolicy.
     *
     * @return parameters used for NoisyNextBestPolicy.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.put("initialExplorationNoise", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("minExplorationNoise", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("explorationNoiseDecay", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for NoisyNextBestPolicy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - initialExplorationNoise: initial exploration noise for NoisyNextBestPolicy. Default value 1.<br>
     *     - minExplorationNoise: minimum exploration noise for NoisyNextBestPolicy. Default value 0.2.<br>
     *     - explorationNoiseDecay: decay factor for exploration noise. Default value 0.999.<br>
     *
     * @param params parameters used for NoisyNextBestPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("initialExplorationNoise")) initialExplorationNoise = params.getValueAsDouble("initialExplorationNoise");
        if (params.hasParam("minExplorationNoise")) minExplorationNoise = params.getValueAsDouble("minExplorationNoise");
        if (params.hasParam("explorationNoiseDecay")) explorationNoiseDecay = params.getValueAsDouble("explorationNoiseDecay");
        explorationNoise = initialExplorationNoise;
    }

    /**
     * Increments policy.
     *
     */
    public void increment() {
        if (explorationNoise > minExplorationNoise) explorationNoise *= explorationNoiseDecay;
    }

    /**
     * Returns action based on policy.
     *
     * @param stateValueSet priority queue containing action values in decreasing order.
     * @return chosen action.
     */
    protected int getAction(TreeSet<ActionValueTuple> stateValueSet) {
        if (stateValueSet.size() > 1 && explorationNoise > random.nextDouble()) stateValueSet.pollLast();
        return stateValueSet.isEmpty() ? -1 : Objects.requireNonNull(stateValueSet.pollLast()).action;
    }

}
