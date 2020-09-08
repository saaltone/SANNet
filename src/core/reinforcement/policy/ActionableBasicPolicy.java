/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.NeuralNetworkException;
import core.reinforcement.Agent;
import core.reinforcement.AgentException;
import core.reinforcement.Environment;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.executablepolicy.ExecutablePolicy;
import utils.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;

/**
 * Class that defines ActionableBasicPolicy.
 *
 */
public class ActionableBasicPolicy implements ActionablePolicy, Serializable {

    private static final long serialVersionUID = -3967136985584564817L;

    /**
     * Reference to environment.
     *
     */
    protected Environment environment;

    /**
     * Reference to FunctionEstimator.
     *
     */
    protected final FunctionEstimator functionEstimator;

    /**
     * If true function estimator is state action value function.
     *
     */
    protected final boolean isStateActionValueFunction;

    /**
     * Reference to executable policy.
     *
     */
    protected final ExecutablePolicy executablePolicy;

    /**
     * If true agent is in learning mode.
     *
     */
    private boolean isLearning = true;

    /**
     * Constructor for ActionableBasicPolicy.
     *
     * @param executablePolicy reference to executable policy.
     * @param functionEstimator reference to FunctionEstimator.
     */
    public ActionableBasicPolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator) {
        this.executablePolicy = executablePolicy;
        this.functionEstimator = functionEstimator;
        isStateActionValueFunction = functionEstimator.isStateActionValue();
    }

    /**
     * Starts ActionableBasicPolicy.
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public void start() throws NeuralNetworkException, MatrixException {
        functionEstimator.start();
    }

    /**
     * Stops ActionableBasicPolicy.
     *
     */
    public void stop() {
        functionEstimator.stop();
    }

    /**
     * Sets reference to environment.
     *
     * @param environment reference to environment.
     */
    public void setEnvironment(Environment environment) {
        this.environment = environment;
    }

    /**
     * Returns reference to environment.
     *
     * @return reference to environment.
     */
    public Environment getEnvironment() {
        return environment;
    }

    /**
     * Set flag if agent is in learning mode.
     *
     * @param isLearning if true agent is in learning mode.
     */
    public void setLearning(boolean isLearning) {
        this.isLearning = isLearning;
    }

    /**
     * Return flag is policy is in learning mode.
     *
     * @return if true agent is in learning mode.
     */
    public boolean isLearning() {
        return isLearning;
    }

    /**
     * Updates executable policy.
     *
     */
    public void update() {
        executablePolicy.increment();
    }

    /**
     * Return state value offset
     *
     * @return state value offset
     */
    protected int getStateValueOffset() {
        return isStateActionValueFunction ? 1 : 0;
    }

    /**
     * Takes action by applying defined executable policy.
     *
     * @param stateTransition state transition.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void act(StateTransition stateTransition) throws NeuralNetworkException, MatrixException {
        Matrix currentPolicyValues = functionEstimator.predict(stateTransition.environmentState.state);
        stateTransition.action = executablePolicy.action(currentPolicyValues, stateTransition.environmentState.availableActions, getStateValueOffset());
        if (isLearning) functionEstimator.add(stateTransition);
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
    }

    /**
     * Returns FunctionEstimator.
     *
     * @return FunctionEstimator.
     */
    public FunctionEstimator getFunctionEstimator() {
        return functionEstimator;
    }

}
