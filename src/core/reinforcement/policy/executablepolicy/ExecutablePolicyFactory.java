/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import core.reinforcement.agent.AgentException;
import utils.configurable.DynamicParamException;

import java.io.Serializable;

/**
 * Factory class that creates executable policy instances.<br>
 *
 */
public class ExecutablePolicyFactory implements Serializable {

    /**
     * Creates executable policy instance of given type with defined parameters.
     *
     * @param executablePolicyType type of executable policy.
     * @param params parameters for executable policy.
     * @return constructed executable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static ExecutablePolicy create(ExecutablePolicyType executablePolicyType, String params) throws DynamicParamException {
        return switch (executablePolicyType) {
            case GREEDY -> new GreedyPolicy();
            case EPSILON_GREEDY -> params == null ? new EpsilonGreedyPolicy() : new EpsilonGreedyPolicy(params);
            case NOISY_NEXT_BEST -> params == null ? new NoisyNextBestPolicy() : new NoisyNextBestPolicy(params);
            case SAMPLED -> params == null ? new SampledPolicy() : new SampledPolicy(params);
            case MCTS -> params == null ? new MCTSPolicy() : new MCTSPolicy(params);
            case ENTROPY_GREEDY -> new EntropyGreedyPolicy();
            case ENTROPY_NOISY_NEXT_BEST -> new EntropyNoisyNextBestPolicy();
        };
    }

    /**
     * Creates executable policy instance of given type with defined parameters.
     *
     * @param executablePolicyType type of executable policy.
     * @return constructed executable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static ExecutablePolicy create(ExecutablePolicyType executablePolicyType) throws DynamicParamException {
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
        if (executablePolicy instanceof EntropyGreedyPolicy) return ExecutablePolicyType.ENTROPY_GREEDY;
        if (executablePolicy instanceof EpsilonGreedyPolicy) return ExecutablePolicyType.EPSILON_GREEDY;
        if (executablePolicy instanceof GreedyPolicy) return ExecutablePolicyType.GREEDY;
        if (executablePolicy instanceof NoisyNextBestPolicy) return ExecutablePolicyType.NOISY_NEXT_BEST;
        if (executablePolicy instanceof SampledPolicy) return ExecutablePolicyType.SAMPLED;
        if (executablePolicy instanceof MCTSPolicy) return ExecutablePolicyType.MCTS;
        if (executablePolicy instanceof EntropyNoisyNextBestPolicy) return ExecutablePolicyType.ENTROPY_NOISY_NEXT_BEST;
        throw new AgentException("Unknown executable policy type");
    }

}
