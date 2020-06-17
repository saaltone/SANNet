/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement.function;

import core.NeuralNetworkException;
import utils.DynamicParamException;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.TreeMap;

/**
 * Interface for function estimators.
 *
 */
public interface FunctionEstimator {

    /**
     * Returns number of actions for function estimator.
     *
     * @return number of actions for function estimator.
     */
    int getNumberOfActions();

    /**
     * Starts function estimator
     *
     * @throws NeuralNetworkException throws exception if starting of function estimator fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    void start() throws NeuralNetworkException, MatrixException;

    /**
     * Stops function estimator.
     *
     */
    void stop();

    /**
     * Returns copy of FunctionEstimator.
     *
     * @return copy of FunctionEstimator.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     */
    FunctionEstimator copy() throws IOException, ClassNotFoundException;

    /**
     * Predicts state values corresponding to a state.
     *
     * @param state state.
     * @return state values corresponding to a state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix predict(Matrix state) throws NeuralNetworkException, MatrixException;

    /**
     * Sets importance sampling weights.
     *
     * @param ISWeights importance sampling weights.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    void setImportanceSamplingWeights(TreeMap<Integer, Double> ISWeights) throws NeuralNetworkException;

    /**
     * Updates (trains) FunctionEstimator.
     *
     * @param states states to be updated.
     * @param stateValues state values to be updated.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void train(LinkedHashMap<Integer, MMatrix> states, LinkedHashMap<Integer, MMatrix> stateValues) throws NeuralNetworkException, MatrixException, DynamicParamException;

    /**
     * Appends parameters of this FunctionEstimator from another FunctionEstimator.
     *
     * @param functionEstimator function estimator used to update current function estimator.
     * @param fullUpdate if true full update is done.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void append(FunctionEstimator functionEstimator, boolean fullUpdate) throws MatrixException;

    /**
     * Returns error of FunctionEstimator.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return error of FunctionEstimator.
     */
    double getError() throws MatrixException;

}
