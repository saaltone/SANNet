package core.reinforcement.policy.executablepolicy;

/**
 * Defines executable policy type.<br>
 * <br>
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
