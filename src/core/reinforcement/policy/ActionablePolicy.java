/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement.policy;

import core.NeuralNetworkException;
import core.reinforcement.Environment;
import core.reinforcement.RLSample;
import core.reinforcement.function.FunctionEstimator;
import utils.matrix.MatrixException;

/**
 * Interface for ActionablePolicy.
 *
 */
public interface ActionablePolicy {

    /**
     * Starts policy FunctionEstimator.
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    void start() throws NeuralNetworkException, MatrixException;

    /**
     * Stops policy FunctionEstimator.
     *
     */
    void stop();

    /**
     * Sets current episode count.
     *
     * @param episodeCount current episode count.
     */
    void setEpisode(int episodeCount);

    /**
     * Sets reference to environment.
     *
     * @param environment reference to environment.
     */
    void setEnvironment(Environment environment);

    /**
     * Returns reference to environment.
     *
     * @return reference to environment.
     */
    Environment getEnvironment();

    /**
     * Takes action by applying defined policy,
     *
     * @param sample sample.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void act(RLSample sample) throws NeuralNetworkException, MatrixException;

    /**
     * Returns policy FunctionEstimator.
     *
     * @return policy FunctionEstimator.
     */
    FunctionEstimator getFunctionEstimator();

}
