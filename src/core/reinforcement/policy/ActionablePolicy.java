/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.reinforcement.Agent;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.executablepolicy.ExecutablePolicy;

/**
 * Class that defines ActionablePolicy.
 *
 */
public class ActionablePolicy extends AbstractPolicy {

    /**
     * Constructor for ActionablePolicy.
     *
     * @param executablePolicy reference to executable policy.
     * @param functionEstimator reference to FunctionEstimator.
     */
    public ActionablePolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator) {
        super(executablePolicy, functionEstimator);
    }

    /**
     * Updates policy.
     *
     */
    public void update() {
    }

    /**
     * Updates policy.
     *
     * @param agent agent.
     */
    public void update(Agent agent) {
    }

}
