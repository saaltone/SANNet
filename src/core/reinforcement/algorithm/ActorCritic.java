/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.Environment;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.policy.updateablepolicy.UpdateableBasicPolicy;
import core.reinforcement.value.StateValueFunctionEstimator;
import utils.DynamicParamException;

/**
 * Class that defines actor critic algorithm.<br>
 *
 */
public class ActorCritic extends AbstractPolicyGradient {

    /**
     * Constructor for ActorCritic
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ActorCritic(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator) throws DynamicParamException {
        super(environment, new UpdateableBasicPolicy(executablePolicyType, policyFunctionEstimator), new StateValueFunctionEstimator(valueFunctionEstimator));
    }

    /**
     * Constructor for ActorCritic
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param params parameters
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ActorCritic(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator, String params) throws DynamicParamException {
        super(environment, new UpdateableBasicPolicy(executablePolicyType, policyFunctionEstimator), new StateValueFunctionEstimator(valueFunctionEstimator), params);
    }

}
