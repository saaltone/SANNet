/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement.policy;

import core.NeuralNetworkException;
import core.reinforcement.RLSample;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.util.TreeMap;

/**
 * Interface for UpdateablePolicy.
 *
 */
public interface UpdateablePolicy extends ActionablePolicy {

    /**
     * Updates policy FunctionEstimator.
     *
     * @param samples samples used for policy FunctionEstimator update.
     * @param hasImportanceSamplingWeights if true samples contain importance sampling weights otherwise false.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void updateFunctionEstimator(TreeMap<Integer, RLSample> samples, boolean hasImportanceSamplingWeights) throws NeuralNetworkException, MatrixException, DynamicParamException;

    /**
     * Sets current value error.
     *
     * @param valueError current value error.
     */
    void setValueError(double valueError);

}
