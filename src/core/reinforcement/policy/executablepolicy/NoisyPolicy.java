/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import core.reinforcement.agent.AgentException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

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
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public NoisyPolicy(String params) throws DynamicParamException, MatrixException {
        super(ExecutablePolicyType.NOISY, params, NoisyPolicy.paramNameTypes);
    }

    /**
     * Initializes default params.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void initializeDefaultParams() throws MatrixException {
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
     * @throws AgentException throws exception if policy fails to choose valid action.
     */
    protected int getAction(TreeSet<ActionValueTuple> stateValueSet) throws AgentException {
        TreeSet<ActionValueTuple> noisedStateValueSet = addNoise(stateValueSet);
        if (noisedStateValueSet.isEmpty()) throw new AgentException("Noisy policy failed to choose valid action.");
        else {
            ActionValueTuple actionValueTuple = noisedStateValueSet.pollLast();
            if (actionValueTuple == null) throw new AgentException("Noisy policy failed to choose valid action.");
            else return actionValueTuple.action();
        }
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

}
