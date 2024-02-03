/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.reinforcement.agent.State;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.IOException;

/**
 * Implements updateable basic policy gradient.<br>
 *
 */
public class UpdateableBasicPolicy extends AbstractUpdateablePolicy {

    /**
     * Log function.
     *
     */
    private final UnaryFunction logFunction = new UnaryFunction(UnaryFunctionType.LOG);

    /**
     * Constructor for updateable basic policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator    reference to function estimator.
     * @param memory               reference to memory.
     * @param params               parameters for updateable basic policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public UpdateableBasicPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, Memory memory, String params) throws DynamicParamException, MatrixException {
        super(executablePolicyType, functionEstimator, memory, params);
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
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param memory                        if true policy will use shared memory otherwise dedicated memory.
     * @return reference to policy.
     * @throws IOException            throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public Policy reference(boolean sharedPolicyFunctionEstimator, Memory memory) throws DynamicParamException, IOException, ClassNotFoundException, MatrixException {
        return new UpdateableBasicPolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(), memory, params);
    }

    /**
     * Returns reference to policy.
     *
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param memory                  if true policy will use shared memory otherwise dedicated memory.
     * @return reference to policy.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public Policy reference(FunctionEstimator policyFunctionEstimator, Memory memory) throws DynamicParamException, MatrixException {
        return new UpdateableBasicPolicy(executablePolicy.getExecutablePolicyType(), policyFunctionEstimator, memory, params);
    }

    /**
     * Returns policy gradient value for update.
     *
     * @param state state.
     * @return policy gradient value.
     */
    protected Matrix getPolicyGradient(State state) throws MatrixException {
        return state.policyValues.apply(logFunction).multiply(state.policyValues.getNewMatrix(state.tdError));
    }

}
