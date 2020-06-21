/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.NeuralNetworkException;
import core.reinforcement.Environment;
import core.reinforcement.RLSample;
import core.reinforcement.function.FunctionEstimator;
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
     * Current policy values estimated by function estimator.
     *
     */
    protected transient Matrix currentPolicyValues;

    /**
     * Reference to FunctionEstimator.
     *
     */
    protected final FunctionEstimator functionEstimator;

    /**
     * Reference to policy.
     *
     */
    protected final Policy policy;

    /**
     * Constructor for policy.
     *
     * @param policy reference to policy.
     * @param functionEstimator reference to FunctionEstimator.
     */
    public ActionableBasicPolicy(Policy policy, FunctionEstimator functionEstimator) {
        this.policy = policy;
        this.functionEstimator = functionEstimator;
    }

    /**
     * Starts policy FunctionEstimator.
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public void start() throws NeuralNetworkException, MatrixException {
        functionEstimator.start();
    }

    /**
     * Stops policy FunctionEstimator.
     *
     */
    public void stop() {
        functionEstimator.stop();
    }

    /**
     * Sets current episode count.
     *
     * @param episodeCount current episode count.
     */
    public void setEpisode(int episodeCount) {
        policy.setEpisode(episodeCount);
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
     * Takes action by applying defined policy,
     *
     * @param sample sample.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void act(RLSample sample) throws NeuralNetworkException, MatrixException {
        sample.state.availableActions = environment.getAvailableActions();
        currentPolicyValues = functionEstimator.predict(sample.state.stateMatrix);
        sample.state.action = policy.action(currentPolicyValues, sample.state.availableActions);
        sample.policyValue = currentPolicyValues.getValue(sample.state.action, 0);
    }

    /**
     * Returns policy FunctionEstimator.
     *
     * @return policy FunctionEstimator.
     */
    public FunctionEstimator getFunctionEstimator() {
        return functionEstimator;
    }

}
