/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.reinforcement.Agent;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.DynamicParamException;

/**
 * Class that defines ActionablePolicy.<br>
 *
 */
public class ActionablePolicy extends AbstractPolicy {

    /**
     * Constructor for ActionablePolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ActionablePolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator) throws DynamicParamException {
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
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @return true if all registered agents are ready to update.
     */
    public boolean readyToUpdate(Agent agent) {
        return true;
    }

    /**
     * Updates policy.
     *
     */
    public void updateFunctionEstimator() {
    }

}
