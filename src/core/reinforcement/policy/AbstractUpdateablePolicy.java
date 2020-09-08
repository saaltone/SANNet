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
public abstract class AbstractUpdateablePolicy extends ActionableBasicPolicy implements ActionablePolicy {

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
     * Returns action with potential state action value offset.
     *
     * @param action action.
     * @return updated action.
     */
    protected int getAction(int action) {
        return getStateValueOffset() + action;
    }

    /**
     * Returns advantage.
     *
     * @param stateTransition state transition
     * @return advantage
     */
    protected double getAdvantage(StateTransition stateTransition) {
        return stateTransition.tdTarget - stateTransition.stateValue;
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
            Matrix policyGradient = new DMatrix(getFunctionEstimator().getNumberOfActions() + getStateValueOffset(), 1);
            if (isStateActionValueFunction) policyGradient.setValue(0, 0, stateTransition.tdTarget);
            policyGradient.setValue(getAction(stateTransition.action), 0, -getPolicyGradientValue(stateTransition));
            getFunctionEstimator().store(agent, stateTransition, policyGradient);
        }
        postProcess();
        getFunctionEstimator().update(agent);
    }

    /**
     * Preprocesses policy gradient setting.
     *
     */
    protected abstract void preProcess();

    /**
     * Returns policy gradient value for StateTransition.
     *
     * @param stateTransition state transition.
     * @return policy gradient value for sample.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract double getPolicyGradientValue(StateTransition stateTransition) throws NeuralNetworkException, MatrixException;

    /**
     * Postprocesses policy gradient setting.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    protected abstract void postProcess() throws MatrixException, AgentException;

}
