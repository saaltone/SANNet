/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;

import java.util.Objects;
import java.util.Random;
import java.util.TreeSet;

/**
 * Implements noisy policy.<br>
 * Policy makes a greedy decision after adding noise to action values.<br>
 *
 */
public class NoisyPolicy extends AbstractExecutablePolicy {

    /**
     * Parameter name types for noisy policy.
     *     - noiseAmplitude: amplitude of noise. Default value 0.1.<br>
     *
     */
    private final static String paramNameTypes = "(noiseAmplitude:DOUBLE)";

    /**
     * Random function for noisy policy.
     *
     */
    private final Random random = new Random();

    /**
     * Amplitude of noise.
     *
     */
    private double noiseAmplitude;

    /**
     * Constructor for noisy policy.
     *
     * @param params parameters for Policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public NoisyPolicy(String params) throws DynamicParamException {
        super(ExecutablePolicyType.NOISY, params, NoisyPolicy.paramNameTypes);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        noiseAmplitude = 0.15;
    }

    /**
     * Returns parameters used for noisy policy.
     *
     * @return parameters used for noisy policy.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + NoisyPolicy.paramNameTypes;
    }

    /**
     * Sets parameters used for noisy policy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - noiseAmplitude: amplitude of noise. Default value 0.1.<br>
     *
     * @param params parameters used for noisy policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("noiseAmplitude")) noiseAmplitude = params.getValueAsDouble("noiseAmplitude");
    }

    /**
     * Increments policy.
     *
     */
    public void increment() {
    }

    /**
     * Returns action based on policy.
     *
     * @param stateValueSet priority queue containing action values in decreasing order.
     * @return chosen action.
     */
    protected int getAction(TreeSet<ActionValueTuple> stateValueSet) {
        TreeSet<ActionValueTuple> noisedStateValueSet = addNoise(stateValueSet);
        return noisedStateValueSet.isEmpty() ? -1 : Objects.requireNonNull(noisedStateValueSet.pollLast()).action();
    }

    /**
     * Adds noise to state value set
     *
     * @param stateValueSet state value set
     * @return noised state value set
     */
    private TreeSet<ActionValueTuple> addNoise(TreeSet<ActionValueTuple> stateValueSet) {
        TreeSet<AbstractExecutablePolicy.ActionValueTuple> result = new TreeSet<>(stateValueSet);
        result.clear();
        for (ActionValueTuple actionValueTuple : stateValueSet) {
            double value = actionValueTuple.value();
            value += noiseAmplitude * random.nextGaussian();
            value = Math.min(1, value);
            value = Math.max(-1, value);
            result.add(new ActionValueTuple(actionValueTuple.action(), value));
        }
        return result;
    }

    /**
     * Applies value as double
     *
     * @param actionValueTuple action value tuple
     * @return returned value
     */
    private static double applyAsDouble(ActionValueTuple actionValueTuple) {
        return actionValueTuple.value();
    }

    /**
     * Resets executable policy.
     *
     */
    public void reset() {
    }

}
