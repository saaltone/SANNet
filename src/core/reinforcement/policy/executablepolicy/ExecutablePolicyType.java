package core.reinforcement.policy.executablepolicy;

/**
 * Defines executable policy type.
 * Currently supported types are:
 *     GREEDY,
 *     EPSILON_GREEDY,
 *     NOISY_NEXT_BEST,
 *     SAMPLED,
 *
 */
public enum ExecutablePolicyType {

    GREEDY,
    EPSILON_GREEDY,
    NOISY_NEXT_BEST,
    SAMPLED,

}
