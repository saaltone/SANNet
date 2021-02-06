package core.reinforcement.algorithm;

import core.reinforcement.Environment;
import core.reinforcement.policy.UpdateableMCTSPolicy;
import core.reinforcement.value.StateValueFunctionEstimator;
import utils.DynamicParamException;

/**
 * Class that defines MCTS learning algorithm.
 *
 */
public class MCTSLearning extends AbstractPolicyGradient {

    /**
     * Constructor for MCTSLearning.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param stateValueFunctionEstimator reference to value function.
     */
    public MCTSLearning(Environment environment, UpdateableMCTSPolicy policy, StateValueFunctionEstimator stateValueFunctionEstimator) {
        super(environment, policy, stateValueFunctionEstimator);
    }

    /**
     * Constructor for MCTSLearning.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param stateValueFunctionEstimator reference to value function.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MCTSLearning(Environment environment, UpdateableMCTSPolicy policy, StateValueFunctionEstimator stateValueFunctionEstimator, String params) throws DynamicParamException {
        super(environment, policy, stateValueFunctionEstimator, params);
    }

}
