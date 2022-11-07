/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.AbstractPolicy;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.IOException;
import java.util.TreeSet;

/**
 * Implements abstract updateable policy.<br>
 * Contains common functions fo updateable policies.<br>
 *
 */
public abstract class AbstractUpdateablePolicy extends AbstractPolicy {

    /**
     * Constructor for abstract updateable policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public AbstractUpdateablePolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator);
    }

    /**
     * Constructor for abstract updateable policy.
     *
     * @param executablePolicy executable policy.
     * @param functionEstimator reference to function estimator.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractUpdateablePolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator) throws AgentException, DynamicParamException {
        super(executablePolicy, functionEstimator);
    }

    /**
     * Constructor for abstract updateable policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to function estimator.
     * @param params parameters for abstract updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public AbstractUpdateablePolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, String params) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator, params);
    }

    /**
     * Constructor for abstract updateable policy.
     *
     * @param executablePolicy executable policy.
     * @param functionEstimator reference to function estimator.
     * @param params parameters for AbstractExecutablePolicy.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractUpdateablePolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator, String params) throws AgentException, DynamicParamException {
        super(executablePolicy, functionEstimator, params);
    }

    /**
     * Return true if policy is updateable otherwise false.
     *
     * @return true if policy is updateable otherwise false.
     */
    public boolean isUpdateablePolicy() {
        return true;
    }

    /**
     * Takes action by applying defined executable policy.
     *
     * @param stateTransition state transition.
     * @param alwaysGreedy if true greedy action is always taken.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void act(StateTransition stateTransition, boolean alwaysGreedy) throws NeuralNetworkException, MatrixException {
        super.act(stateTransition, alwaysGreedy);
        if (isLearning()) executablePolicy.add(stateTransition);
    }

    /**
     * Ends episode
     *
     */
    public void endEpisode() {
        executablePolicy.endEpisode();
    }

    /**
     * Resets function estimator.
     *
     */
    public void resetFunctionEstimator() {
        functionEstimator.reset();
    }

    /**
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @throws AgentException throws exception if agent is not registered for function estimator.
     * @return true if all registered agents are ready to update.
     */
    public boolean readyToUpdate(Agent agent) throws AgentException {
        return functionEstimator.readyToUpdate(agent);
    }

    /**
     * Updates function estimator.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException, IOException, ClassNotFoundException {
        TreeSet<StateTransition> sampledStateTransitions = functionEstimator.getSampledStateTransitions();
        if (sampledStateTransitions == null || sampledStateTransitions.isEmpty()) {
            functionEstimator.abortUpdate();
            return;
        }

        for (StateTransition stateTransition : sampledStateTransitions) functionEstimator.store(stateTransition, getPolicyValues(stateTransition));
        postProcess();

        functionEstimator.update();
    }

    /**
     * Returns policy values.
     *
     * @param stateTransition state transition.
     * @return policy values.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    private Matrix getPolicyValues(StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
        if (isStateActionValueFunction()) {
            Matrix stateValue = new DMatrix(1, 1);
            stateValue.setValue(0, 0, stateTransition.tdTarget);
            Matrix policyValues = new DMatrix(functionEstimator.getNumberOfActions(), 1);
            policyValues.setValue(stateTransition.action, 0, getPolicyValue(stateTransition));
            return new JMatrix(new Matrix[] {stateValue, policyValues}, true);
        }
        else {
            Matrix policyValues = new DMatrix(functionEstimator.getNumberOfActions(), 1);
            policyValues.setValue(stateTransition.action, 0, getPolicyValue(stateTransition));
            return policyValues;
        }
    }

    /**
     * Returns policy value for state transition.
     *
     * @param stateTransition state transition.
     * @return policy gradient value for sample.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract double getPolicyValue(StateTransition stateTransition) throws NeuralNetworkException, MatrixException;

    /**
     * Postprocesses policy gradient update.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    protected void postProcess() throws MatrixException, AgentException, NeuralNetworkException, IOException, DynamicParamException, ClassNotFoundException {
    }

    /**
     * Appends parameters to this policy from another policy.
     *
     * @param policy policy used to update current policy.
     * @param tau tau which controls contribution of other policy.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void append(Policy policy, double tau) throws MatrixException, AgentException, NeuralNetworkException, IOException, DynamicParamException, ClassNotFoundException {
        functionEstimator.append(policy.getFunctionEstimator(), tau);
    }

}
