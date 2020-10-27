/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.NeuralNetworkException;
import core.reinforcement.Agent;
import core.reinforcement.AgentException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.executablepolicy.ExecutablePolicy;
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
     * @param executablePolicy reference to policy.
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
        if (isLearning()) executablePolicy.update();
        executablePolicy.finish();
    }

    /**
     * Updates policy.
     *
     * @param agent agent.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    public void update(Agent agent) throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException {
        if (getFunctionEstimator().sampledSetEmpty()) return;
        TreeSet<StateTransition> stateTransitions = getFunctionEstimator().getSampledStateTransitions();

        preProcess();
        for (StateTransition stateTransition : stateTransitions) {
            Matrix policyValues = new DMatrix(getFunctionEstimator().getNumberOfActions() + getStateValueOffset(), 1);
            if (isStateActionValueFunction) policyValues.setValue(0, 0, stateTransition.tdTarget);
            policyValues.setValue(getAction(stateTransition.action), 0, getPolicyValue(stateTransition));
            getFunctionEstimator().store(agent, stateTransition, policyValues);
        }
        postProcess();

        getFunctionEstimator().update(agent);
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
