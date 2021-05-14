/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.NeuralNetworkException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.StateTransition;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Class that defines QTargetValueFunctionEstimator (Q value function with target function estimator).<br>
 *
 */
public class QTargetValueFunctionEstimator extends AbstractValueFunctionEstimator {

    /**
     * Constructor for QTargetValueFunctionEstimator.
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public QTargetValueFunctionEstimator(FunctionEstimator functionEstimator) throws IOException, ClassNotFoundException, DynamicParamException, MatrixException {
        super(functionEstimator.getNumberOfActions(), functionEstimator);
        functionEstimator.setTargetFunctionEstimator();
    }

    /**
     * Constructor for QTargetValueFunctionEstimator.
     *
     * @param functionEstimator reference to value FunctionEstimator.
     * @param params parameters for QTargetValueFunctionEstimator.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public QTargetValueFunctionEstimator(FunctionEstimator functionEstimator, String params) throws IOException, ClassNotFoundException, DynamicParamException, MatrixException {
        super(functionEstimator.getNumberOfActions(), functionEstimator, params);
        functionEstimator.setTargetFunctionEstimator();
    }

    /**
     * Returns target value based on next state. Uses target action with maximal value as defined by target FunctionEstimator.
     *
     * @param nextStateTransition next state transition.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getTargetValue(StateTransition nextStateTransition) throws NeuralNetworkException, MatrixException {
        return functionEstimator.getTargetFunctionEstimator().predict(nextStateTransition.environmentState.state()).getValue(functionEstimator.argmax(functionEstimator.predict(nextStateTransition.environmentState.state()), nextStateTransition.environmentState.availableActions()), 0);
    }

}
