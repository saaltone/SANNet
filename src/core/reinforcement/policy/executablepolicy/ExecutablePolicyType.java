/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

/**
 * Defines executable policy type.
 *
 */
public enum ExecutablePolicyType {

    /**
     * Greedy policy
     *
     */
    GREEDY,

    /**
     * Epsilon greedy policy
     *
     */
    EPSILON_GREEDY,

    /**
     * Noisy next best policy
     *
     */
    NOISY_NEXT_BEST,

    /**
     * Sampled i.e. stochastic policy
     *
     */
    SAMPLED,

}
