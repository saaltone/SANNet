/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.network.NeuralNetworkException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.agent.State;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.MCTSPolicy;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements updateable MCTS policy.<br>
 *
 */
public class UpdateableMCTSPolicy extends AbstractUpdateablePolicy {

    /**
     * Constructor for updateable MCTS policy.
     *
     * @param functionEstimator reference to function estimator.
     * @param mctsPolicy        reference to MCTS policy.
     * @param memory            reference to memory.
     * @param params            parameters for UpdateableBasicPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public UpdateableMCTSPolicy(FunctionEstimator functionEstimator, MCTSPolicy mctsPolicy, Memory memory, String params) throws DynamicParamException, MatrixException {
        super (mctsPolicy, functionEstimator, memory, params);
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
        return new UpdateableMCTSPolicy(sharedPolicyFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(), sharedPolicyFunctionEstimator ? (MCTSPolicy)executablePolicy : new MCTSPolicy(), memory, params);
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
        return new UpdateableMCTSPolicy(policyFunctionEstimator, (MCTSPolicy)executablePolicy, memory, params);
    }

    /**
     * Returns policy value for update.
     *
     * @param state state.
     * @return policy value.
     */
    protected Matrix getPolicyGradient(State state) throws MatrixException, NeuralNetworkException {
        Matrix policyValues = state.policyValues.getNewMatrix();
        policyValues.setValue(state.action, 0, 0, state.policyValue * Math.log(getValues(state).getValue(state.action, 0, 0) + 10E-8));
        return policyValues;
    }

}
