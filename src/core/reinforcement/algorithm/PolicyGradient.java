/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.Environment;
import core.reinforcement.policy.Policy;
import core.reinforcement.value.ValueFunction;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Class that defines generic policy gradient algorithm.<br>
 * Can be used to implement Actor Critic and REINFORCE algorithms.<br>
 *
 */
public class PolicyGradient extends AbstractPolicyGradient {

    /**
     * Constructor for PolicyGradient.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to valueFunction.
     */
    public PolicyGradient(Environment environment, Policy policy, ValueFunction valueFunction) {
        super(environment, policy, valueFunction);
    }

    /**
     * Constructor for PolicyGradient.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to valueFunction.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public PolicyGradient(Environment environment, Policy policy, ValueFunction valueFunction, String params) throws DynamicParamException {
        super(environment, policy, valueFunction, params);
    }

    /**
     * Returns reference to algorithm.
     *
     * @return reference to algorithm.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public PolicyGradient reference() throws DynamicParamException {
        return new PolicyGradient(getEnvironment(), policy, valueFunction, getParams());
    }

    /**
     * Returns reference to algorithm.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws NeuralNetworkException throws exception if optimizer is of an unknown type.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public PolicyGradient reference(boolean sharedPolicyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, NeuralNetworkException, IOException, DynamicParamException, ClassNotFoundException, AgentException {
        return new PolicyGradient(getEnvironment(), policy.reference(sharedPolicyFunctionEstimator, sharedMemory), valueFunction.reference(sharedValueFunctionEstimator, sharedMemory), getParams());
    }

}
