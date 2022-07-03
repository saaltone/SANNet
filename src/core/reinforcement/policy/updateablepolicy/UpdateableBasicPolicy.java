/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.AgentException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements vanilla updateable basic policy.<br>
 *
 */
public class UpdateableBasicPolicy extends AbstractUpdateablePolicy {

    /**
     * Constructor for updateable basic policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public UpdateableBasicPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator);
    }

    /**
     * Constructor for updateable basic policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to function estimator.
     * @param params parameters for updateable basic policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public UpdateableBasicPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, String params) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator, params);
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
     * @return reference to policy.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference() throws DynamicParamException, AgentException, MatrixException, IOException, ClassNotFoundException {
        return new UpdateableBasicPolicy(executablePolicy.getExecutablePolicyType(), functionEstimator.reference(), params);
    }

    /**
     * Returns reference to policy.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to policy.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference(boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws DynamicParamException, AgentException, MatrixException, IOException, ClassNotFoundException {
        return new UpdateableBasicPolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? functionEstimator : functionEstimator.reference(sharedMemory), params);
    }

    /**
     * Returns policy gradient value for update.
     *
     * @param stateTransition state transition.
     * @return policy gradient value.
     */
    protected double getPolicyValue(StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
        Matrix currentPolicyValues = functionEstimator.predict(stateTransition);
        double currentPolicyValue = currentPolicyValues.getValue(stateTransition.action, 0);
        return Math.log(currentPolicyValue) * stateTransition.advantage;
    }

}
