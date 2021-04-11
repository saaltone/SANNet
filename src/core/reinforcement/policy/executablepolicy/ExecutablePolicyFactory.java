package core.reinforcement.policy.executablepolicy;

import core.reinforcement.AgentException;
import utils.DynamicParamException;

import java.io.Serializable;

/**
 * Factory class that creates executable policy instances.<br>
 *
 */
public class ExecutablePolicyFactory implements Serializable {

    private static final long serialVersionUID = 4044647047494437807L;

    /**
     * Creates executable policy instance of given type with defined parameters.
     *
     * @param executablePolicyType type of executable policy.
     * @param params parameters for executable policy.
     * @return constructed executable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if creation of executable policy fails.
     */
    public static ExecutablePolicy create(ExecutablePolicyType executablePolicyType, String params) throws DynamicParamException, AgentException {
        switch (executablePolicyType) {
            case GREEDY:
                return new GreedyPolicy();
            case EPSILON_GREEDY:
                return params == null ? new EpsilonGreedyPolicy() : new EpsilonGreedyPolicy(params);
            case NOISY_NEXT_BEST:
                return params == null ? new NoisyNextBestPolicy() : new NoisyNextBestPolicy(params);
            case SAMPLED:
                return params == null ? new SampledPolicy() : new SampledPolicy(params);
        }
        throw new AgentException("Creation of executable policy failed.");
    }

    /**
     * Creates executable policy instance of given type with defined parameters.
     *
     * @param executablePolicyType type of executable policy.
     * @return constructed executable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if creation of executable policy fails.
     */
    public static ExecutablePolicy create(ExecutablePolicyType executablePolicyType) throws DynamicParamException, AgentException {
        return create(executablePolicyType, null);
    }

    /**
     * Returns type of executable policy.
     *
     * @param executablePolicy executable policy.
     * @return type of executable policy.
     * @throws AgentException throws exception is executable policy is of unknown type.
     */
    public static ExecutablePolicyType getExecutablePolicyType(ExecutablePolicy executablePolicy) throws AgentException {
        if (executablePolicy instanceof EpsilonGreedyPolicy) return ExecutablePolicyType.EPSILON_GREEDY;
        if (executablePolicy instanceof GreedyPolicy) return ExecutablePolicyType.GREEDY;
        if (executablePolicy instanceof NoisyNextBestPolicy) return ExecutablePolicyType.NOISY_NEXT_BEST;
        if (executablePolicy instanceof SampledPolicy) return ExecutablePolicyType.SAMPLED;
        throw new AgentException("Unknown executable policy type");
    }

}
