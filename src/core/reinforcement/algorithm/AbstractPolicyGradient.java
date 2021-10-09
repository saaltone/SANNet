/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.DeepAgent;
import core.reinforcement.agent.Environment;
import core.reinforcement.policy.Policy;
import core.reinforcement.value.ValueFunction;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Class that defines policy gradient algorithms.<br>
 *
 */
public abstract class AbstractPolicyGradient extends DeepAgent {

    /**
     * Constructor for AbstractPolicyGradient.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to value function.
     */
    public AbstractPolicyGradient(Environment environment, Policy policy, ValueFunction valueFunction) {
        super(environment, policy, valueFunction);
    }

    /**
     * Constructor for AbstractPolicyGradient.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to value function.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractPolicyGradient(Environment environment, Policy policy, ValueFunction valueFunction, String params) throws DynamicParamException {
        super(environment, policy, valueFunction, params);
    }

    /**
     * Updates policy and value functions of agent.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    protected void updateFunctionEstimator() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException {
        if(policy.readyToUpdate(this) && valueFunction.readyToUpdate(this)) {
            valueFunction.sample();
            if (!updateValuePerEpisode) valueFunction.update();
            valueFunction.updateFunctionEstimator();
            policy.updateFunctionEstimator();
            valueFunction.resetFunctionEstimator();
            policy.resetFunctionEstimator();
        }
    }

    /**
     * Returns reference to AbstractPolicyGradient algorithm.
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
    public abstract AbstractPolicyGradient reference(boolean sharedPolicyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, NeuralNetworkException, IOException, DynamicParamException, ClassNotFoundException, AgentException;

}
