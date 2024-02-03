/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.*;
import core.reinforcement.policy.Policy;
import core.reinforcement.value.ValueFunction;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.TreeSet;

/**
 * Implements abstract policy gradient functionality.<br>
 *
 */
public abstract class AbstractPolicyGradient extends DeepAgent {

    /**
     * Constructor for abstract policy gradient.
     *
     * @param stateSynchronization reference to state synchronization.
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to value function.
     */
    public AbstractPolicyGradient(StateSynchronization stateSynchronization, Environment environment, Policy policy, ValueFunction valueFunction) {
        super(stateSynchronization, environment, policy, valueFunction);
    }

    /**
     * Constructor for abstract policy gradient.
     *
     * @param stateSynchronization reference to state synchronization.
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to value function.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractPolicyGradient(StateSynchronization stateSynchronization, Environment environment, Policy policy, ValueFunction valueFunction, String params) throws DynamicParamException {
        super(stateSynchronization, environment, policy, valueFunction, params);
        policy.getFunctionEstimator().setEnableImportanceSamplingWeights(false);
    }

    /**
     * Updates policy and value functions of agent.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    protected void updateFunctionEstimator() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException, IOException, ClassNotFoundException {
        TreeSet<State> sampledStates = null;
        if(valueFunction.readyToUpdate(this)) {
            valueFunction.sample();
            if (!updateValuePerEpisode) valueFunction.update();
            sampledStates = valueFunction.updateFunctionEstimator();
            valueFunction.resetFunctionEstimator();
        }
        if(policy.readyToUpdate(this) && sampledStates != null) {
            policy.updateFunctionEstimator(sampledStates);
            policy.resetFunctionEstimator();
        }
    }

    /**
     * Returns reference to abstract policy gradient algorithm.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public abstract AbstractPolicyGradient reference(boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException;

    /**
     * Returns reference to abstract policy gradient algorithm.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedValueFunctionEstimator if true shared value function estimator is used otherwise new policy function estimator is created.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public abstract AbstractPolicyGradient reference(boolean sharedPolicyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, IOException, DynamicParamException, ClassNotFoundException, AgentException;

}
