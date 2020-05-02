/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement.value;

import core.NeuralNetworkException;
import core.reinforcement.RLSample;
import core.reinforcement.State;
import core.reinforcement.function.FunctionEstimator;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.util.TreeMap;

/**
 * Interface for ValueFunction.
 *
 */
public interface ValueFunction {

    /**
     * Starts FunctionEstimator
     *
     * @throws NeuralNetworkException throws exception if starting of value function fails.
     */
    void start() throws NeuralNetworkException;

    /**
     * Stops FunctionEstimator
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
     * Return target value for sample originating from next state.
     *
     * @param nextState next state.
     * @return target value of state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    double getTargetValue(State nextState) throws NeuralNetworkException, MatrixException;

    /**
     * Returns TD target of sample.
     *
     * @param state state.
     * @return TD target of state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    double getTDTarget(State state) throws NeuralNetworkException, MatrixException;

    /**
     * Updates TD target of sample.
     *
     * @param sample sample.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void updateTDTarget(RLSample sample) throws NeuralNetworkException, MatrixException;

    /**
     * Returns FunctionEstimator.
     *
     * @return FunctionEstimator.
     */
    FunctionEstimator getFunctionEstimator();

    /**
     * Updates FunctionEstimator.
     *
     * @param samples samples used to update FunctionEstimator.
     * @param hasImportanceSamplingWeights if true samples contain importance sampling weights otherwise false.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void updateFunctionEstimator(TreeMap<Integer, RLSample> samples, boolean hasImportanceSamplingWeights) throws NeuralNetworkException, MatrixException, DynamicParamException;

    /**
     * Updates target value FunctionEstimator.<br>
     * If update cycle is greater than 0 makes full update every update cycle episodes else applies smooth update with update rate tau.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void updateTargetFunctionEstimator() throws MatrixException;

    /**
     * Sets current estimator version.
     *
     * @param estimatorVersion current estimator version.
     */
    void setEstimatorVersion(int estimatorVersion);

    /**
     * Returns current value error.
     *
     * @return current value error.
     */
    double getValueError();

}
