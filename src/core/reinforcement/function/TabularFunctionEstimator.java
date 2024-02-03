/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.function;

import core.optimization.*;
import core.reinforcement.memory.Memory;
import core.reinforcement.agent.State;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Implements tabular based state action function estimator.<br>
 * Reference for polynomial learning rate: <a href="https://www.jmlr.org/papers/volume5/evendar03a/evendar03a.pdf">...</a> <br>
 *
 */
public class TabularFunctionEstimator extends AbstractFunctionEstimator {

    /**
     * Parameter name types for tabular function estimator.
     *     - optimizerName: name of optimizer for tabular function estimator. Default value "Adam".<br>
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *
     */
    private final static String paramNameTypes = "(optimizerName:String), " +
            "(learningRate:DOUBLE)";

    /**
     * Hash map to store state values pairs.
     *
     */
    private HashMap<Matrix, Matrix> stateValues = new HashMap<>();

    /**
     * Intermediate map for state value pairs for function update.
     *
     */
    private final HashMap<State, Matrix> stateValueMap = new HashMap<>();

    /**
     * Optimizer for tabular function estimator.
     *
     */
    private Optimizer optimizer;

    /**
     * Constructor for tabular function estimator.
     *
     * @param memory memory reference.
     * @param numberOfStates number of states for tabular function estimator
     * @param numberOfActions number of actions for tabular function estimator
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public TabularFunctionEstimator(Memory memory, int numberOfStates, int numberOfActions) throws DynamicParamException {
        super (memory, numberOfStates, numberOfActions, false);
    }

    /**
     * Constructor for tabular function estimator.
     *
     * @param memory memory reference.
     * @param numberOfStates number of states for tabular function estimator
     * @param numberOfActions number of actions for tabular function estimator
     * @param params params for tabular function estimator
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public TabularFunctionEstimator(Memory memory, int numberOfStates, int numberOfActions, String params) throws DynamicParamException {
        super (memory, numberOfStates, numberOfActions, false, params);
    }

    /**
     * Constructor for tabular function estimator.
     *
     * @param memory memory reference.
     * @param numberOfStates number of states for tabular function estimator
     * @param numberOfActions number of actions for tabular function estimator
     * @param stateValues state values inherited for tabular function estimator.
     * @param params params for tabular function estimator
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public TabularFunctionEstimator(Memory memory, int numberOfStates, int numberOfActions, HashMap<Matrix, Matrix> stateValues, String params) throws DynamicParamException {
        super (memory, numberOfStates, numberOfActions, false, params);
        this.stateValues = stateValues;
    }

    /**
     * Initializes default params.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void initializeDefaultParams() throws DynamicParamException {
        optimizer = new Adam();
    }

    /**
     * Returns parameters used for tabular function estimator.
     *
     * @return parameters used for tabular function estimator.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + TabularFunctionEstimator.paramNameTypes;
    }

    /**
     * Sets parameters used for tabular function estimator.<br>
     * <br>
     * Supported parameters are:<br>
     *     - optimizerName: name of optimizer for tabular function estimator. Default value "Adam".<br>
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *
     * @param params parameters used for tabular function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("optimizerName")) {
            String optimizerName = params.getValueAsString("optimizerName");
            double learningRate = 0.001;
            if (params.hasParam("learningRate")) learningRate = params.getValueAsDouble("learningRate");
            optimizer = OptimizerFactory.create(optimizerName, "learningRate = " + learningRate);
        }
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
     * Returns reference to function estimator.
     *
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator reference() throws DynamicParamException {
        return new TabularFunctionEstimator(getMemory().reference(), getNumberOfStates(), getNumberOfActions(), getParams());
    }

    /**
     * Returns reference to function estimator.
     *
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator reference(boolean sharedMemory) throws DynamicParamException {
        return new TabularFunctionEstimator(sharedMemory ? getMemory() : getMemory().reference(), getNumberOfStates(), getNumberOfActions(), getParams());
    }

    /**
     * Returns reference to function estimator.
     *
     * @param memory reference to memory.
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator reference(Memory memory) throws DynamicParamException {
        return new TabularFunctionEstimator(memory, getNumberOfStates(), getNumberOfActions(), getParams());
    }

    /**
     * Returns shallow copy of tabular function estimator.
     *
     * @return shallow copy of tabular function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator copy() throws DynamicParamException {
        return new TabularFunctionEstimator(memory, getNumberOfStates(), getNumberOfActions(), stateValues, getParams());
    }

    /**
     * Returns state values corresponding to a state or if state does not exist creates and returns new state value matrix.
     *
     * @param state state
     * @return state values corresponding to a state
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix getStateValue(Matrix state) throws MatrixException {
        for (Map.Entry<Matrix, Matrix> entry : stateValues.entrySet()) {
            if (state.equals(entry.getKey())) return entry.getValue();
        }

        Matrix stateValue = new DMatrix(numberOfActions, 1, 1, Initialization.RANDOM);
        stateValues.put(state, stateValue);
        return stateValue;
    }

    /**
     * Resets tabular function estimator.
     *
     */
    public void reset() {
        super.reset();
        stateValueMap.clear();
    }

    /**
     * Returns (predicts) state value corresponding to a state as stored by tabular function estimator.
     *
     * @param state state
     * @return state value corresponding to a state
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix predictPolicyValues(State state) throws MatrixException {
        return getStateValue(state.environmentState.state());
    }

    /**
     * Predicts target policy values corresponding to a state.
     *
     * @param state state.
     * @return policy values corresponding to a state.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix predictTargetPolicyValues(State state) throws MatrixException {
        return predictPolicyValues(state);
    }

    /**
     * Predicts state action values corresponding to a state.
     *
     * @param state state.
     * @return state action values corresponding to a state.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix predictStateActionValues(State state) throws MatrixException {
        return predictPolicyValues(state);
    }

    /**
     * Predicts target state action values corresponding to a state.
     *
     * @param state state.
     * @return state action values corresponding to a state.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix predictTargetStateActionValues(State state) throws MatrixException {
        return predictPolicyValues(state);
    }

    /**
     * Stores policy state values pair.
     *
     * @param state state.
     * @param values values.
     */
    public void storePolicyValues(State state, Matrix values) {
        stateValueMap.put(state, values);
    }

    /**
     * Stores state action values pair.
     *
     * @param state state.
     * @param values values.
     */
    public void storeStateActionValues(State state, Matrix values) {
        storePolicyValues(state, values);
    }

    /**
     * Updates (trains) tabular function estimator.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void update() throws MatrixException, DynamicParamException {
        HashMap<Matrix, Matrix> stateErrors = new HashMap<>();
        for (Map.Entry<State, Matrix> entry : stateValueMap.entrySet()) {
            Matrix stateValue = predictPolicyValues(entry.getKey());
            Matrix error = stateValue.subtract(entry.getValue());
            Matrix stateError = stateErrors.get(stateValue);
            if (stateError == null) stateErrors.put(stateValue, error);
            else stateError.addBy(error);
        }
        for (Map.Entry<Matrix, Matrix> entry : stateErrors.entrySet()) {
            Matrix stateValue = entry.getKey();
            Matrix stateError = entry.getValue();
            optimizer.optimize(stateValue, stateError.divide(stateValueMap.size()));
        }

        stateValueMap.clear();

        // Allows other threads to get execution time.
        try {
            TimeUnit.NANOSECONDS.sleep(1);
        }
        catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(exception);
        }

        updateComplete();
        reset();
    }

    /**
     * Sets if importance sampling weights are applied.
     *
     * @param applyImportanceSamplingWeights if true importance sampling weights are applied otherwise not.
     */
    public void setEnableImportanceSamplingWeights(boolean applyImportanceSamplingWeights) {
    }

    /**
     * Appends from function estimator.
     *
     * @param functionEstimator function estimator.
     */
    public void append(FunctionEstimator functionEstimator) {
    }

}
