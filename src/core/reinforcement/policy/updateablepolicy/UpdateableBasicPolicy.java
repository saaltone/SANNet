/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.AgentException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Class that defines vanilla UpdateableBasicPolicy.<br>
 *
 */
public class UpdateableBasicPolicy extends AbstractUpdateablePolicy {

    /**
     * Parameter name types for UpdateableBasicPolicy.
     *     - applyEntropy: if true entropy is applied when policy is updated. Default value true.<br>
     *     - entropyCoefficient: co-efficient factor for entropy. Default value 0.01.<br>
     *
     */
    private final static String paramNameTypes = "(applyEntropy:BOOLEAN), " +
            "(entropyCoefficient:DOUBLE)";

    /**
     * If true entropy is applied when policy is updated.
     *
     */
    protected boolean applyEntropy;

    /**
     * Entropy coefficient for policy gradient.
     *
     */
    protected double entropyCoefficient;

    /**
     * Constructor for UpdateableBasicPolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public UpdateableBasicPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator);
    }

    /**
     * Constructor for UpdateableBasicPolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for UpdateableBasicPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public UpdateableBasicPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, String params) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        applyEntropy = true;
        entropyCoefficient = 0.01;
    }

    /**
     * Returns parameters used for UpdateableBasicPolicy.
     *
     * @return parameters used for UpdateableBasicPolicy.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + UpdateableBasicPolicy.paramNameTypes;
    }

    /**
     * Sets parameters used for UpdateableBasicPolicy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - applyEntropy: if true entropy is applied when policy is updated. Default value true.<br>
     *     - entropyCoefficient: co-efficient factor for entropy. Default value 0.01.<br>
     *
     * @param params parameters used for UpdateableBasicPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("applyEntropy")) applyEntropy = params.getValueAsBoolean("applyEntropy");
        if (params.hasParam("entropyCoefficient")) entropyCoefficient = params.getValueAsDouble("entropyCoefficient");
    }

    /**
     * Returns reference to policy.
     *
     * @return reference to policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference() throws DynamicParamException, AgentException {
        return new UpdateableBasicPolicy(executablePolicy.getExecutablePolicyType(), functionEstimator, params);
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
        return new UpdateableBasicPolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? functionEstimator : functionEstimator.reference(sharedMemory), params);
    }

    /**
     * Returns policy gradient value for update.
     *
     * @param stateTransition state transition.
     * @return policy gradient value.
     */
    protected double getPolicyValue(StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
        Matrix currentPolicyValues = functionEstimator.predict(stateTransition.environmentState.state());
        double epsilon = 10E-8;
        double currentPolicyValue = currentPolicyValues.getValue(stateTransition.action, 0) + epsilon;
        return -(currentPolicyValue * Math.log(currentPolicyValue) * (stateTransition.advantage + (applyEntropy ? entropyCoefficient * currentPolicyValues.entropy(true) : 0)));
    }

}
