/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.reinforcement.AgentException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.DynamicParamException;

/**
 * Class that defines ActionablePolicy.
 *
 */
public class ActionablePolicy extends AbstractPolicy {

    /**
     * Constructor for ActionablePolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if creation of executable policy fails.
     */
    public ActionablePolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator);
    }

    /**
     * Updates policy.
     *
     */
    public void update() {
    }

    /**
     * Resets FunctionEstimator.
     *
     */
    public void resetFunctionEstimator() {
    }

    /**
     * Updates policy.
     *
     */
    public void updateFunctionEstimator() {
    }

}
