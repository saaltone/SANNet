/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.NeuralNetworkException;
import core.reinforcement.State;
import core.reinforcement.function.FunctionEstimator;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

/**
 * Class that defines QValueFunctionEstimator (Q value function with function estimator).
 *
 */
public class QValueFunctionEstimator extends AbstractValueFunctionEstimator {

    /**
     * Constructor for QValueFunctionEstimator.
     *
     * @param functionEstimator reference to FunctionEstimator.
     */
    public QValueFunctionEstimator(FunctionEstimator functionEstimator) {
        super(functionEstimator);
    }

    /**
     * Constructor for QValueFunctionEstimator.
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for QValueFunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public QValueFunctionEstimator(FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super(functionEstimator, params);
    }

    /**
     * Return target value for state based on it's next state. Uses max function to choose target value.
     *
     * @param nextState next state.
     * @return target value for state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getTargetValue(State nextState) throws NeuralNetworkException, MatrixException {
        return max(functionEstimator.predict(nextState.stateMatrix));
    }

    /**
     * Updates target value FunctionEstimator.
     *
     */
    public void updateTargetFunctionEstimator() {}



}
