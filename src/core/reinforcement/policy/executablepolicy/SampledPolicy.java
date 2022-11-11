/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;

import java.util.Objects;
import java.util.Random;
import java.util.TreeSet;

/**
 * Implements sampled policy.<br>
 *
 */
public class SampledPolicy extends AbstractExecutablePolicy {

    /**
     * Parameter name types for sampled policy.
     *     - thresholdInitial: initial threshold value for sampling randomness. Default value 1.<br>
     *     - thresholdMin: lowest value for threshold. Default value 0.2.<br>
     *     - thresholdDecay: decay rate of threshold. Default value 0.999.<br>
     *
     */
    private final static String paramNameTypes = "(thresholdInitial:DOUBLE), " +
            "(thresholdMin:DOUBLE), " +
            "(thresholdDecay:DOUBLE)";

    /**
     * Random function for sampled policy.
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
     * Default constructor for sampled policy.
     *
     */
    public SampledPolicy() {
        super(ExecutablePolicyType.SAMPLED);
    }

    /**
     * Constructor for sampled policy.
     *
     * @param params parameters for sampled policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public SampledPolicy(String params) throws DynamicParamException {
        super(ExecutablePolicyType.SAMPLED, params, SampledPolicy.paramNameTypes);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        thresholdInitial = 1;
        thresholdMin = 0.2;
        thresholdDecay = 0.999;
        thresholdCurrent = thresholdInitial;
    }

    /**
     * Returns parameters used for sampled policy.
     *
     * @return parameters used for sampled policy.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + SampledPolicy.paramNameTypes;
    }

    /**
     * Sets parameters used for sampled policy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - thresholdInitial: initial threshold value for sampling randomness. Default value 1.<br>
     *     - thresholdMin: lowest value for threshold. Default value 0.2.<br>
     *     - thresholdDecay: decay rate of threshold. Default value 0.999.<br>
     *
     * @param params parameters used for sampled policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
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
     * @param stateValueSet priority queue containing action values in decreasing order.
     * @return chosen action.
     */
    protected int getAction(TreeSet<ActionValueTuple> stateValueSet) {
        double lowValue = stateValueSet.first().value();
        double highValue = stateValueSet.last().value();
        double thresholdValue = highValue - (highValue - lowValue) * thresholdCurrent * random.nextDouble();
        while (!stateValueSet.isEmpty()) {
            ActionValueTuple actionValueTuple = stateValueSet.pollFirst();
            if (Objects.requireNonNull(actionValueTuple).value() >= thresholdValue) return actionValueTuple.action();
        }
        return -1;
    }

    /**
     * Resets executable policy.
     *
     */
    public void reset() {
    }

}
