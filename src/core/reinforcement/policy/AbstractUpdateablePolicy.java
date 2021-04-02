/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.NeuralNetworkException;
import core.reinforcement.AgentException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.executablepolicy.ExecutablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.TreeSet;

/**
 * Class that defines AbstractUpdateablePolicy. Contains common functions fo updateable policies.
 *
 */
public abstract class AbstractUpdateablePolicy extends AbstractPolicy {

    /**
     * Constructor for AbstractUpdateablePolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if creation of executable policy fails.
     */
    public AbstractUpdateablePolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator);
    }

    /**
     * Constructor for AbstractUpdateablePolicy.
     *
     * @param executablePolicy executable policy.
     * @param functionEstimator reference to FunctionEstimator.
     */
    public AbstractUpdateablePolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator) {
        super(executablePolicy, functionEstimator);
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
        if (isLearning()) executablePolicy.record(stateTransition);
    }

    /**
     * Updates policy.
     *
     */
    public void update() {
        executablePolicy.finish(isLearning());
    }

    /**
     * Resets FunctionEstimator.
     *
     */
    public void resetFunctionEstimator() {
        functionEstimator.reset();
    }

    /**
     * Updates policy.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    public void updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException {
        TreeSet<StateTransition> sampledStateTransitions = functionEstimator.getSampledStateTransitions();
        if (sampledStateTransitions == null || sampledStateTransitions.isEmpty()) return;

        preProcess();
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
        Matrix policyValues = new DMatrix(functionEstimator.getNumberOfActions() + isStateActionValueFunction(), 1);
        if (isStateActionValueFunction) policyValues.setValue(0, 0, stateTransition.tdTarget);
        policyValues.setValue(getAction(stateTransition.action), 0, getPolicyValue(stateTransition));
        return policyValues;
    }

    /**
     * Preprocesses policy gradient update.
     *
     */
    protected void preProcess() {
    }

    /**
     * Returns policy value for StateTransition.
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
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    protected void postProcess() throws MatrixException, AgentException {
    }

}
