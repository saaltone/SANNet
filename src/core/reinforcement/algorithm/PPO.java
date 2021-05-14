/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.Environment;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.policy.updateablepolicy.UpdateableProximalPolicy;
import core.reinforcement.value.StateValueFunctionEstimator;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Class that defines Proximal Policy Optimization (PPO).<br>
 *
 */
public class PPO extends AbstractPolicyGradient {

    /**
     * Constructor for PPO
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public PPO(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator) throws ClassNotFoundException, MatrixException, DynamicParamException, IOException {
        super(environment, new UpdateableProximalPolicy(executablePolicyType, policyFunctionEstimator), new StateValueFunctionEstimator(valueFunctionEstimator));
    }

    /**
     * Constructor for PPO
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param params parameters for agent.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public PPO(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator, String params) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        super(environment, new UpdateableProximalPolicy(executablePolicyType, policyFunctionEstimator), new StateValueFunctionEstimator(valueFunctionEstimator), params);
    }

}