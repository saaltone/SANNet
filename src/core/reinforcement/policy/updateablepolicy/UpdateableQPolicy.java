/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.State;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements updateable Q policy gradient.<br>
 *
 */
public class UpdateableQPolicy extends AbstractUpdateablePolicy {

    /**
     * Reference to value function estimator.
     *
     */
    private final FunctionEstimator valueFunctionEstimator;

    /**
     * Constructor for updateable basic policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to function estimator.
     * @param valueFunctionEstimator    reference to value function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException        throws exception if state action value function is applied to non-updateable policy.
     */
    public UpdateableQPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, FunctionEstimator valueFunctionEstimator) throws DynamicParamException, AgentException {
        this(executablePolicyType, functionEstimator, valueFunctionEstimator, null);
    }

    /**
     * Constructor for updateable basic policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param params parameters for updateable basic policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public UpdateableQPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, FunctionEstimator valueFunctionEstimator, String params) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator, params);
        this.valueFunctionEstimator = valueFunctionEstimator;
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
    }

    /**
     * Returns reference to policy.
     *
     * @param valueFunctionEstimator reference to value function estimator.
     * @return reference to policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference(FunctionEstimator valueFunctionEstimator) throws DynamicParamException, AgentException {
        return new UpdateableQPolicy(executablePolicy.getExecutablePolicyType(), getFunctionEstimator(), valueFunctionEstimator, params);
    }

    /**
     * Returns reference to policy.
     *
     * @param valueFunctionEstimator reference to value function estimator.
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to policy.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference(FunctionEstimator valueFunctionEstimator, boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws DynamicParamException, AgentException, MatrixException, IOException, ClassNotFoundException {
        return new UpdateableQPolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(sharedMemory), valueFunctionEstimator, params);
    }

    /**
     * Returns reference to policy.
     *
     * @param valueFunctionEstimator        reference to value function estimator.
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param memory                        reference to memory.
     * @return reference to policy.
     * @throws IOException            throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws AgentException         throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference(FunctionEstimator valueFunctionEstimator, boolean sharedPolicyFunctionEstimator, Memory memory) throws DynamicParamException, IOException, ClassNotFoundException, AgentException, MatrixException {
        return new UpdateableQPolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(memory), valueFunctionEstimator, params);
    }

    /**
     * Returns policy gradient value for update.
     *
     * @param state state.
     * @return policy gradient value.
     */
    protected double getPolicyValue(State state) throws MatrixException, NeuralNetworkException {
        return -valueFunctionEstimator.predictStateActionValues(state).getValue(getFunctionEstimator().predictTargetPolicyValues(state).argmax()[0], 0, 0);
    }

}
