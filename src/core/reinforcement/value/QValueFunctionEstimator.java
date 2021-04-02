/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.NeuralNetworkException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
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
        super(functionEstimator.getNumberOfActions(), functionEstimator);
    }

    /**
     * Returns target value based on next state. Uses max value of next state.
     *
     * @param nextStateTransition next state transition.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getTargetValue(StateTransition nextStateTransition) throws NeuralNetworkException, MatrixException {
        return functionEstimator.max(functionEstimator.predict(nextStateTransition.environmentState.state), nextStateTransition.environmentState.availableActions);
    }

}
