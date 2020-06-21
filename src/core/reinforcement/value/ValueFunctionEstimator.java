/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.NeuralNetworkException;
import core.reinforcement.State;
import core.reinforcement.function.FunctionEstimator;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.util.HashMap;

/**
 * Class that defines ValueFunctionEstimator (value function with function estimator).
 *
 */
public class ValueFunctionEstimator extends AbstractValueFunctionEstimator {

    /**
     * If true uses bootstrapping for value function target value.
     *
     */
    private boolean bootstrap = true;

    /**
     * Constructor for ValueFunctionEstimator
     *
     * @param functionEstimator reference to FunctionEstimator.
     */
    public ValueFunctionEstimator(FunctionEstimator functionEstimator) {
        super(functionEstimator);
    }

    /**
     * Constructor for ValueFunctionEstimator
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ValueFunctionEstimator(FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super(functionEstimator, params);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for ValueFunctionEstimator.
     *
     * @return parameters used for ValueFunctionEstimator.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.put("bootstrap", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for ValueFunctionEstimator.<br>
     * <br>
     * Supported parameters are:<br>
     *     - bootstrap: if true uses bootstrapping for value function target value. Default value true.<br>
     *
     * @param params parameters used for ValueFunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("bootstrap")) bootstrap = params.getValueAsBoolean("bootstrap");
    }

    /**
     * Returns target value matching current sample per next state.
     *
     * @param nextState next state.
     * @return target value of state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getTargetValue(State nextState) throws NeuralNetworkException, MatrixException {
        return bootstrap ? functionEstimator.predict(nextState.stateMatrix).getValue(getAction(nextState), 0) : nextState.sample.stateValues.getValue(getAction(nextState), 0);
    }

    /**
     * Updates target value FunctionEstimator.
     *
     */
    public void updateTargetFunctionEstimator() {}

}
