/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.agent.State;
import core.reinforcement.policy.updateablepolicy.UpdateableSoftQPolicy;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.IOException;

/**
 * Implements soft Q value function.<br>
 *
 */
public class SoftQValueFunction extends QTargetValueFunction {

    /**
     * Reference to current policy.
     *
     */
    private UpdateableSoftQPolicy updateableSoftQPolicy;

    /**
     * Constructor for soft Q value function.
     *
     * @param functionEstimator       reference to value FunctionEstimator.
     * @param params                  parameters for QTargetValueFunctionEstimator.
     */
    public SoftQValueFunction(FunctionEstimator functionEstimator, String params) {
        this(functionEstimator, null, true, true, params);
    }

    /**
     * Constructor for soft Q value function.
     *
     * @param functionEstimator  reference to value function estimator.
     * @param functionEstimator2 reference to second value function estimator.
     * @param dualFunctionEstimation if true soft Q value function estimator has dual function estimator.
     * @param usesTargetValueFunctionEstimator if true uses target value function estimator.
     * @param params             parameters for target Q value function estimator.
     */
    protected SoftQValueFunction(FunctionEstimator functionEstimator, FunctionEstimator functionEstimator2, boolean dualFunctionEstimation, boolean usesTargetValueFunctionEstimator, String params) {
        super(functionEstimator, functionEstimator2, dualFunctionEstimation, usesTargetValueFunctionEstimator, params);
    }

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @return reference to value function.
     * @throws IOException            throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws MatrixException        throws exception if neural network has less output than actions.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new SoftQValueFunction(sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(), sharedValueFunctionEstimator ? getFunctionEstimator2() : null, dualFunctionEstimation, isUsingTargetValueFunctionEstimator(), getParams());
    }

    /**
     * Sets updateable soft Q policy.
     *
     * @param updateableSoftQPolicy updateable soft Q policy.
     */
    public void setUpdateableSoftQPolicy(UpdateableSoftQPolicy updateableSoftQPolicy) {
        this.updateableSoftQPolicy = updateableSoftQPolicy;
    }

    /**
     * Returns policy function estimator.
     *
     * @return policy function estimator.
     */
    private UpdateableSoftQPolicy getUpdateableSoftQPolicy() {
        return updateableSoftQPolicy;
    }

    /**
     * Returns target value based on next state. Uses target action with maximal value as defined by target FunctionEstimator.
     *
     * @param nextState next state.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public double getTargetValue(State nextState) throws NeuralNetworkException, MatrixException, DynamicParamException {
        int targetAction = getTargetAction(nextState);
        double qTargetValue = getTargetValues(nextState, isUsingTargetValueFunctionEstimator()).getValue(targetAction, 0, 0);
        double policyValue = getUpdateableSoftQPolicy().getFunctionEstimator().predictPolicyValues(nextState).getValue(targetAction, 0, 0);
        double softAlpha = getUpdateableSoftQPolicy().getSoftQAlphaMatrix().getValue(0, 0, 0);
        return policyValue * (qTargetValue - softAlpha * Math.log(policyValue));
    }

    /**
     * Returns target action based on next state.
     *
     * @param nextState next state.
     * @return target action based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    private int getTargetAction(State nextState) throws NeuralNetworkException, MatrixException {
        return getUpdateableSoftQPolicy().getFunctionEstimator().sample(getUpdateableSoftQPolicy().getFunctionEstimator().predictPolicyValues(nextState), nextState.environmentState.availableActions());
    }

}
