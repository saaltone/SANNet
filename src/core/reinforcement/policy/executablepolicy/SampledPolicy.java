/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import utils.DynamicParam;
import utils.DynamicParamException;

import java.util.Objects;
import java.util.Random;
import java.util.TreeSet;

/**
 * Class that defines SampledPolicy.<br>
 *
 */
public class SampledPolicy extends AbstractExecutablePolicy {

    /**
     * Parameter name types for SampledPolicy.
     *     - thresholdInitial: initial threshold value for sampling randomness. Default value 1.<br>
     *     - thresholdMin: lowest value for threshold. Default value 0.2.<br>
     *     - thresholdDecay: decay rate of threshold. Default value 0.999.<br>
     *
     */
    private final static String paramNameTypes = "(thresholdInitial:DOUBLE), " +
            "(thresholdMin:DOUBLE), " +
            "(thresholdDecay:DOUBLE)";

    /**
     * Executable policy type.
     *
     */
    private final ExecutablePolicyType executablePolicyType = ExecutablePolicyType.SAMPLED;

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
        super(params, SampledPolicy.paramNameTypes);
    }

    /**
     * Returns parameters used for SampledPolicy.
     *
     * @return parameters used for SampledPolicy.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + SampledPolicy.paramNameTypes;
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
        return stateValueSet.first().action();
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
