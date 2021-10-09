/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import utils.DynamicParamException;

/**
 * Class that defines StateValueFunctionEstimator (state value function with function estimator).<br>
 *
 */
public class StateValueFunctionEstimator extends AbstractValueFunctionEstimator {

    /**
     * Constructor for StateValueFunctionEstimator
     *
     * @param functionEstimator reference to FunctionEstimator.
     */
    public StateValueFunctionEstimator(FunctionEstimator functionEstimator) {
        super(1, functionEstimator);
    }

    /**
     * Constructor for StateValueFunctionEstimator
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public StateValueFunctionEstimator(FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super(1, functionEstimator, params);
    }

    /**
     * Returns reference to value function.
     *
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ValueFunction reference() throws DynamicParamException {
        return new StateValueFunctionEstimator(functionEstimator, getParams());
    }

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if optimizer is of an unknown type.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator, boolean sharedMemory) throws DynamicParamException, NeuralNetworkException {
        return new StateValueFunctionEstimator(sharedValueFunctionEstimator ? functionEstimator : functionEstimator.reference(sharedMemory), getParams());
    }

    /**
     * Returns function index. Always 0.
     *
     * @param stateTransition state transition.
     * @return updated action.
     */
    protected int getFunctionIndex(StateTransition stateTransition) {
        return 0;
    }

    /**
     * Returns target value based on next state.
     *
     * @param nextStateTransition next state transition.
     * @return target value based on next state
     */
    public double getTargetValue(StateTransition nextStateTransition) {
        return nextStateTransition.tdTarget;
    }

}
