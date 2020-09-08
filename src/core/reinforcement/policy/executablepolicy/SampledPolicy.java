/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import utils.DynamicParam;
import utils.DynamicParamException;

import java.util.HashMap;
import java.util.PriorityQueue;
import java.util.Random;

/**
 * Class that defined SampledPolicy which chooses action based on weighted random cumulative value.
 *
 */
public class SampledPolicy extends AbstractExecutablePolicy {

    /**
     * Random function for SampledPolicy.
     *
     */
    private final Random random = new Random();

    /**
     * Current threshold
     *
     */
    private double thresholdCurrent;

    /**
     * Initial threshold
     *
     */
    private double thresholdInitial = 1;

    /**
     * Minimum threshold
     *
     */
    private double thresholdMin = 0.2;

    /**
     * Threshold decay
     *
     */
    private double thresholdDecay = 0.999;

    /**
     * Default constructor for SampledPolicy.
     *
     */
    public SampledPolicy() {
        thresholdCurrent = thresholdInitial;
    }

    /**
     * Constructor for SampledPolicy.
     *
     * @param params parameters for SampledPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public SampledPolicy(String params) throws DynamicParamException {
        super(params);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for SampledPolicy.
     *
     * @return parameters used for SampledPolicy.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.put("thresholdInitial", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("thresholdMin", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("thresholdDecay", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for SampledPolicy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - thresholdInitial: initial threshold value for sampling randomness. Default value 1.<br>
     *     - thresholdMin: lowest value for threshold. Default value 0.2.<br>
     *     - thresholdDecay: decay rate of threshold. Default value 0.999.<br>
     *
     * @param params parameters used for SampledPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("thresholdInitial")) thresholdInitial = params.getValueAsDouble("thresholdInitial");
        if (params.hasParam("thresholdMin")) thresholdMin = params.getValueAsDouble("thresholdMin");
        if (params.hasParam("thresholdDecay")) thresholdDecay = params.getValueAsDouble("thresholdDecay");
        thresholdCurrent = thresholdInitial;
    }

    /**
     * Increments policy.
     *
     */
    public void increment() {
        if (thresholdCurrent > thresholdMin) thresholdCurrent *= thresholdDecay;
    }

    /**
     * Returns action based on policy.
     *
     * @param stateValuePriorityQueue priority queue containing action values in decreasing order.
     * @param cumulativeValue cumulative value of actions.
     * @return chosen action.
     */
    protected int getAction(PriorityQueue<ActionValueTuple> stateValuePriorityQueue, double cumulativeValue) {
        double thresholdValue = cumulativeValue * thresholdCurrent * random.nextDouble();
        double currentCumulativeValue = 0;
        while (!stateValuePriorityQueue.isEmpty()) {
            ActionValueTuple actionValueTuple = stateValuePriorityQueue.poll();
            currentCumulativeValue += actionValueTuple.value;
            if (thresholdValue <= currentCumulativeValue) return actionValueTuple.action;
        }
        return -1;
    }

}
