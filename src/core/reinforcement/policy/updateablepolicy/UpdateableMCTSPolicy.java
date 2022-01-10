/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.AgentException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.MCTSPolicy;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

/**
 * Class that defines updateable MCTS policy.<br>
 *
 */
public class UpdateableMCTSPolicy extends AbstractUpdateablePolicy {

    /**
     * Constructor for UpdateableMCTSPolicy.
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public UpdateableMCTSPolicy(FunctionEstimator functionEstimator) throws AgentException, DynamicParamException {
        super (new MCTSPolicy(), functionEstimator);
    }

    /**
     * Constructor for UpdateableMCTSPolicy.
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @param mctsPolicy reference to MCTS policy.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public UpdateableMCTSPolicy(FunctionEstimator functionEstimator, MCTSPolicy mctsPolicy) throws AgentException, DynamicParamException {
        super (mctsPolicy, functionEstimator);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
    }

    /**
     * Constructor for UpdateableMCTSPolicy.
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for UpdateableBasicPolicy.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public UpdateableMCTSPolicy(FunctionEstimator functionEstimator, String params) throws AgentException, DynamicParamException {
        super (new MCTSPolicy(), functionEstimator, params);
    }

    /**
     * Constructor for UpdateableMCTSPolicy.
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @param mctsPolicy reference to MCTS policy.
     * @param params parameters for UpdateableBasicPolicy.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public UpdateableMCTSPolicy(FunctionEstimator functionEstimator, MCTSPolicy mctsPolicy, String params) throws AgentException, DynamicParamException {
        super (mctsPolicy, functionEstimator, params);
    }

    /**
     * Returns reference to policy.
     *
     * @return reference to policy.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Policy reference() throws AgentException, DynamicParamException {
        return new UpdateableMCTSPolicy(functionEstimator, new MCTSPolicy(), params);
    }

    /**
     * Returns reference to policy.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference(boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws DynamicParamException, AgentException {
        return new UpdateableMCTSPolicy(sharedPolicyFunctionEstimator ? functionEstimator : functionEstimator.reference(sharedMemory), sharedPolicyFunctionEstimator ? (MCTSPolicy)executablePolicy : new MCTSPolicy(), params);
    }

    /**
     * Returns policy value for update.
     *
     * @param stateTransition state transition.
     * @return policy value.
     */
    protected double getPolicyValue(StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
        return -stateTransition.value * Math.log(functionEstimator.predict(stateTransition.environmentState.state()).getValue(stateTransition.action, 0) + 10E-6);
    }

}
