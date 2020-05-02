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
import utils.Sample;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.LinkedHashMap;
import java.util.TreeMap;

/**
 * Class that defines AbstractValueFunctionEstimator.
 *
 */
public abstract class AbstractValueFunctionEstimator extends AbstractValueFunction {

    /**
     * Reference to FunctionEstimator.
     *
     */
    protected FunctionEstimator functionEstimator;

    /**
     * Constructor for AbstractValueFunctionEstimator
     *
     * @param functionEstimator reference to FunctionEstimator.
     */
    public AbstractValueFunctionEstimator(FunctionEstimator functionEstimator) {
        super(functionEstimator.getNumberOfActions());
        this.functionEstimator = functionEstimator;
    }

    /**
     * Constructor for AbstractValueFunctionEstimator
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractValueFunctionEstimator(FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super(functionEstimator.getNumberOfActions(), params);
        this.functionEstimator = functionEstimator;
    }

    /**
     * Starts FunctionEstimator
     *
     * @throws NeuralNetworkException throws exception if starting of value FunctionEstimator fails.
     */
    public void start() throws NeuralNetworkException {
        functionEstimator.start();
    }

    /**
     * Stops FunctionEstimator
     *
     */
    public void stop() {
        functionEstimator.stop();
    }

    /**
     * Updates baseline value for sample.
     *
     * @param sample sample.
     */
    protected void updateBaseline(RLSample sample) {
        sample.baseline = sample.getValue(getAction(sample.state));
    }

    /**
     * Returns values for state.
     *
     * @param state state.
     * @return values for state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix getValues(State state) throws NeuralNetworkException, MatrixException {
        return functionEstimator.predict(state.stateMatrix);
    }

    /**
     * Updates FunctionEstimator.
     *
     * @param samples samples used to update FunctionEstimator.
     * @param hasImportanceSamplingWeights if true samples contain importance sampling weights otherwise false.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void updateFunctionEstimator(TreeMap<Integer, RLSample> samples, boolean hasImportanceSamplingWeights) throws NeuralNetworkException, MatrixException, DynamicParamException {
        super.updateFunctionEstimator(samples, hasImportanceSamplingWeights);
        LinkedHashMap<Integer, Sample> states = new LinkedHashMap<>();
        LinkedHashMap<Integer, Sample> stateValues = new LinkedHashMap<>();
        TreeMap<Integer, Double> importanceSamplingWeights = hasImportanceSamplingWeights ? new TreeMap<>() : null;
        int sampleIndex = 0;
        for (RLSample sample : samples.values()) {
            states.put(sampleIndex, new Sample(sample.state.stateMatrix));
            sample.setValue(getAction(sample.state), sample.tdTarget - sample.baseline);
            stateValues.put(sampleIndex, new Sample(sample.stateValues));
            if (importanceSamplingWeights != null) importanceSamplingWeights.put(sampleIndex, sample.importanceSamplingWeight);
            sampleIndex++;
        }
        getFunctionEstimator().setImportanceSamplingWeights(importanceSamplingWeights);
        getFunctionEstimator().train(states, stateValues);
        updateTargetFunctionEstimator();
    }

    /**
     * Calculates max of state values.
     *
     * @param stateValues state values.
     * @return max value.
     */
    protected double max(Matrix stateValues) {
        return stateValues.getValue(argmax(stateValues), 0);
    }

    /**
     * Calculates argmax of state values.
     *
     * @param stateValues state values.
     * @return argmax value.
     */
    protected int argmax(Matrix stateValues) {
        int action = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int row = 0; row < stateValues.getRows(); row++) {
            double actionValue = stateValues.getValue(row, 0);
            if (maxValue < actionValue || maxValue == Double.NEGATIVE_INFINITY) {
                maxValue =  actionValue;
                action = row;
            }
        }
        return action;
    }

    /**
     * Returns FunctionEstimator.
     *
     * @return FunctionEstimator.
     */
    public FunctionEstimator getFunctionEstimator() {
        return functionEstimator;
    }

    /**
     * Returns current value error.
     *
     * @return current value error.
     */
    public double getValueError() {
        return functionEstimator.getError();
    }

}
