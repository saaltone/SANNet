/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement.algorithm;

import core.NeuralNetworkException;
import core.reinforcement.*;
import core.reinforcement.policy.UpdateablePolicy;
import core.reinforcement.value.ValueFunction;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.util.TreeMap;

/**
 * Class that defines policy gradient algorithms.
 *
 */
public abstract class AbstractPolicyGradient extends DeepAgent {

    private final UpdateablePolicy updateablePolicy;

    /**
     * Constructor for Policy Gradient.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param buffer reference to buffer.
     * @param valueFunction reference to value function.
     */
    public AbstractPolicyGradient(Environment environment, UpdateablePolicy policy, Buffer buffer, ValueFunction valueFunction) {
        super(environment, policy, buffer, valueFunction);
        this.updateablePolicy = policy;
    }

    /**
     * Constructor for Policy Gradient.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param buffer reference to buffer.
     * @param valueFunction reference to value function.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractPolicyGradient(Environment environment, UpdateablePolicy policy, Buffer buffer, ValueFunction valueFunction, String params) throws DynamicParamException {
        super(environment, policy, buffer, valueFunction, params);
        this.updateablePolicy = policy;
    }

    /**
     * Updates policy and value functions of agent.
     *
     * @param samples samples used to update function estimator.
     * @param hasImportanceSamplingWeights if true samples contain importance sampling weights otherwise false.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void updateAgent(TreeMap<Integer, RLSample> samples, boolean hasImportanceSamplingWeights) throws NeuralNetworkException, MatrixException, DynamicParamException {
        valueFunction.updateFunctionEstimator(samples, hasImportanceSamplingWeights);
        updateablePolicy.setValueError(valueFunction.getValueError());
        updateablePolicy.updateFunctionEstimator(samples, hasImportanceSamplingWeights);
    }

    /**
     * Updates estimator version.
     *
     * @param estimatorVersion estimator version.
     */
    protected void updateEstimatorVersion(int estimatorVersion) {
        valueFunction.setEstimatorVersion(estimatorVersion);
        updateablePolicy.setEstimatorVersion(estimatorVersion);
    }

}
