/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.function;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.TreeMap;

/**
 * Class that defines tabular state action function estimator.
 *
 */
public class TabularFunctionEstimator implements FunctionEstimator, Serializable {

    private static final long serialVersionUID = 4324050226101969329L;

    /**
     * Hash map to store state (action) values.
     *
     */
    private HashMap<Matrix, Matrix> stateValues = new HashMap<>();

    /**
     * Number of actions for function.
     *
     */
    private final int numberOfActions;

    /**
     * Learning rate for training.
     *
     */
    private double learningRate = 0.1;

    /**
     * Constructor for TabularFunctionEstimator.
     *
     * @param numberOfActions number of actions for TabularFunctionEstimator
     */
    public TabularFunctionEstimator(int numberOfActions) {
        this.numberOfActions = numberOfActions;
    }

    /**
     * Constructor for TabularFunctionEstimator.
     *
     * @param numberOfActions number of actions for TabularFunctionEstimator
     * @param params parameters for function
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public TabularFunctionEstimator(int numberOfActions, String params) throws DynamicParamException {
        this.numberOfActions = numberOfActions;
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Constructor for TabularFunctionEstimator.
     *
     * @param numberOfActions number of actions for TabularFunctionEstimator
     * @param stateValues state values inherited for TabularFunctionEstimator.
     */
    public TabularFunctionEstimator(int numberOfActions, HashMap<Matrix, Matrix> stateValues) {
        this.numberOfActions = numberOfActions;
        this.stateValues = stateValues;
    }

    /**
     * Returns parameters used for TabularFunctionEstimator.
     *
     * @return parameters used for TabularFunctionEstimator.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("learningRate", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for TabularFunctionEstimator.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for tabular function updates. Default value 0.1.<br>
     *
     * @param params parameters used for TabularFunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("learningRate")) learningRate = params.getValueAsDouble("learningRate");
    }

    /**
     * Returns number of actions for TabularFunctionEstimator.
     *
     * @return number of actions for TabularFunctionEstimator.
     */
    public int getNumberOfActions() {
        return numberOfActions;
    }

    /**
     * Not used.
     *
     */
    public void start() {
    }

    /**
     * Not used.
     *
     */
    public void stop() {
    }

    /**
     * Sets state values map for TabularFunctionEstimator.
     *
     * @param stateValues state values map
     */
    public void setStateValues(HashMap<Matrix, Matrix> stateValues) {
        this.stateValues = stateValues;
    }

    /**
     * Returns state values map for TabularFunctionEstimator.
     *
     * @return state values map
     */
    public HashMap<Matrix, Matrix> getStateValues() {
        return stateValues;
    }

    /**
     * Returns state values corresponding to a state or if state does not exists creates and returns new state value matrix.
     *
     * @param state state
     * @return state values corresponding to a state
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix getStateValue(Matrix state) throws MatrixException {
        for (Matrix existingState : stateValues.keySet()) {
            if (state.equals(existingState)) return stateValues.get(existingState);
        }
        Matrix stateValue = new DMatrix(numberOfActions, 1);
        stateValues.put(state, stateValue);
        return stateValue;
    }

    /**
     * Returns shallow copy of TabularFunctionEstimator.
     *
     * @return shallow copy of TabularFunctionEstimator.
     */
    public FunctionEstimator copy() {
        return new TabularFunctionEstimator(getNumberOfActions(), getStateValues());
    }

    /**
     * Returns (predicts) state value corresponding to a state as stored by TabularFunctionEstimator.
     *
     * @param state state
     * @return state value corresponding to a state
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix predict(Matrix state) throws MatrixException {
        return getStateValue(state).copy();
    }

    /**
     * Not used.
     *
     * @param ISWeights importance sampling weights.
     */
    public void setImportanceSamplingWeights(TreeMap<Integer, Double> ISWeights) {}

    /**
     * Updates (trains) TabularFunctionEstimator.
     *
     * @param states states to be updated.
     * @param stateValues state values to be updated.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void train(LinkedHashMap<Integer, MMatrix> states, LinkedHashMap<Integer, MMatrix> stateValues) throws MatrixException {
        for (Integer index : states.keySet()) {
            // currentStateValue: Q(s,a) stored by TabularFunctionEstimator
            // targetStateValue: reward + gamma * targetValue per updated TD target
            // Q(s,a) = Q(s,a) + learningRate * (reward + gamma * targetValue - Q(s,a))
            Matrix currentStateValue = getStateValue(states.get(index).get(0));
            Matrix targetStateValue = stateValues.get(index).get(0);
            currentStateValue.add(targetStateValue.subtract(currentStateValue).multiply(learningRate), currentStateValue);
        }
        // Allows other threads to get execution time.
        try {
            Thread.sleep(1);
        } catch (InterruptedException e) {}
    }

    /**
     * Updates function estimator to match current state values.
     *
     * @param functionEstimator estimator function used to update this function.
     * @param fullUpdate if true full update is done.
     */
    public void append(FunctionEstimator functionEstimator, boolean fullUpdate) {
         ((TabularFunctionEstimator) functionEstimator).setStateValues(stateValues);
    }

    /**
     * Returns error of FunctionEstimator.
     *
     * @return error of FunctionEstimator.
     */
    public double getError() {
        return 0;
    }

}
