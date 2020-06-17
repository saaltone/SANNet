/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement.value;

import core.NeuralNetworkException;
import core.reinforcement.State;
import core.reinforcement.function.FunctionEstimator;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.HashMap;

/**
 * Class that defines QTargetValueFunctionEstimator (Q value function with target function estimator).
 *
 */
public class QTargetValueFunctionEstimator extends AbstractValueFunctionEstimator {

    /**
     * Target value FunctionEstimator.
     *
     */
    private final FunctionEstimator targetValueFunctionEstimator;

    /**
     * Update cycle (in episodes) for target FunctionEstimator. If update cycle is zero then applies smooth updates with update rate tau.
     *
     */
    private int updateCycle = 0;

    /**
     * Constructor for QTargetValueFunctionEstimator.
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     */
    public QTargetValueFunctionEstimator(FunctionEstimator functionEstimator) throws IOException, ClassNotFoundException {
        super(functionEstimator);
        targetValueFunctionEstimator = functionEstimator.copy();
    }

    /**
     * Constructor for QTargetValueFunctionEstimator.
     *
     * @param functionEstimator reference to value FunctionEstimator.
     * @param params parameters for QTargetValueFunctionEstimator.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public QTargetValueFunctionEstimator(FunctionEstimator functionEstimator, String params) throws IOException, ClassNotFoundException, DynamicParamException {
        super(functionEstimator, params);
        setParams(new DynamicParam(params, getParamDefs()));
        targetValueFunctionEstimator = functionEstimator.copy();
    }

    /**
     * Returns parameters used for QTargetValueFunctionEstimator.
     *
     * @return parameters used for QTargetValueFunctionEstimator.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.put("updateCycle", DynamicParam.ParamType.INT);
        return paramDefs;
    }

    /**
     * Sets parameters used for QTargetValueFunctionEstimator.<br>
     * <br>
     * Supported parameters are:<br>
     *     - updateCycle; update cycle for target function (assumes full update). Default value 0 i.e. applies smooth update.<br>
     *
     * @param params parameters used for QTargetValueFunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("updateCycle")) updateCycle = params.getValueAsInteger("updateCycle");
    }

    /**
     * Starts FunctionEstimator
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public void start() throws NeuralNetworkException, MatrixException {
        super.start();
        targetValueFunctionEstimator.start();
    }

    /**
     * Stops FunctionEstimator
     *
     */
    public void stop() {
        super.stop();
        targetValueFunctionEstimator.stop();
    }

    /**
     * Return target value for state based on it's next state.<br>
     * Uses value function to calculate target action and target value function to calculate target value given target action.<br>
     *
     * @param nextState next state.
     * @return target value for state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getTargetValue(State nextState) throws NeuralNetworkException, MatrixException {
        return targetValueFunctionEstimator.predict(nextState.stateMatrix).getValue(argmax(functionEstimator.predict(nextState.stateMatrix)), 0);
    }

    /**
     * Updates target value FunctionEstimator.<br>
     * if update cycle is greater than 0 makes full update every update cycle episodes else applies smooth update with update rate tau.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void updateTargetFunctionEstimator() throws MatrixException {
        if (updateCycle == 0) targetValueFunctionEstimator.append(functionEstimator, false);
        else if (episodeCount % updateCycle == 0) targetValueFunctionEstimator.append(functionEstimator, true);
    }

}
