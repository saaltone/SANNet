/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.function;

import core.network.NeuralNetworkException;
import core.optimization.*;
import core.reinforcement.agent.AgentException;
import core.reinforcement.memory.Memory;
import core.reinforcement.agent.StateTransition;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Implements tabular based state action function estimator.<br>
 * Reference for polynomial learning rate: https://www.jmlr.org/papers/volume5/evendar03a/evendar03a.pdf <br>
 *
 */
public class TabularFunctionEstimator extends AbstractFunctionEstimator {

    /**
     * Parameter name types for tabular function estimator.
     *     - optimizerName: name of optimizer for tabular function estimator. Default value "Adadelta".<br>
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
     * Intermediate map for state transition value pairs for function update.
     *
     */
    private final HashMap<StateTransition, Matrix> stateTransitionValueMap = new HashMap<>();

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
        super.initializeDefaultParams();
        optimizer = new Adadelta();
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
     *     - optimizerName: name of optimizer for tabular function estimator. Default value "Adadelta".<br>
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
     * Checks if function estimator is started.
     *
     * @return true if function estimator is started otherwise false.
     */
    public boolean isStarted() {
        return true;
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
     * Sets state values map for tabular function estimator.
     *
     * @param newStateValues new state values map
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private void setStateValues(HashMap<Matrix, Matrix> newStateValues) throws MatrixException {
        stateValues.clear();
        for (Map.Entry<Matrix, Matrix> entry : newStateValues.entrySet()) {
            Matrix currentState = entry.getKey();
            Matrix stateValue = entry.getValue();
            stateValues.put(currentState.copy(), stateValue.copy());
        }
    }

    /**
     * Returns state values map of tabular function estimator.
     *
     * @return state values map
     */
    public HashMap<Matrix, Matrix> getStateValues() {
        return stateValues;
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
            Matrix currentState = entry.getKey();
            Matrix stateValue = entry.getValue();
            if (state.equals(currentState)) return stateValue;
        }
        Matrix stateValue = new DMatrix(numberOfActions, 1, Initialization.RANDOM);
        stateValues.put(state.copy(), stateValue);
        return stateValue;
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
     * Resets tabular function estimator.
     *
     */
    public void reset() {
        super.reset();
        stateTransitionValueMap.clear();
    }

    /**
     * Reinitializes tabular function estimator.
     *
     */
    public void reinitialize() {
        this.reset();
    }

    /**
     * Returns (predicts) state value corresponding to a state as stored by tabular function estimator.
     *
     * @param stateTransition state
     * @return state value corresponding to a state
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix predictPolicyValues(StateTransition stateTransition) throws MatrixException {
        return getStateValue(stateTransition.environmentState.state());
    }

    /**
     * Predicts state action values corresponding to a state.
     *
     * @param stateTransition state.
     * @return state action values corresponding to a state.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix predictStateActionValues(StateTransition stateTransition) throws MatrixException {
        return predictPolicyValues(stateTransition);
    }

    /**
     * Stores policy state transition values pair.
     *
     * @param stateTransition state transition.
     * @param values values.
     */
    public void storePolicyValues(StateTransition stateTransition, Matrix values) {
        stateTransitionValueMap.put(stateTransition, values);
    }

    /**
     * Stores state action state transition values pair.
     *
     * @param stateTransition state transition.
     * @param values values.
     */
    public void storeStateActionValues(StateTransition stateTransition, Matrix values) {
        storePolicyValues(stateTransition, values);
    }

    /**
     * Updates (trains) tabular function estimator.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void update() throws MatrixException, AgentException, DynamicParamException, NeuralNetworkException, IOException, ClassNotFoundException {
        HashMap<Matrix, Matrix> stateErrors = new HashMap<>();
        for (Map.Entry<StateTransition, Matrix> entry : stateTransitionValueMap.entrySet()) {
            StateTransition stateTransition = entry.getKey();
            Matrix stateValueEntry = entry.getValue();
            Matrix stateValue = predictPolicyValues(stateTransition);
            Matrix error = stateValue.subtract(stateValueEntry);
            Matrix stateError = stateErrors.get(stateValue);
            if (stateError == null) stateErrors.put(stateValue, error);
            else stateError.add(error, stateError);
        }
        for (Map.Entry<Matrix, Matrix> entry : stateErrors.entrySet()) {
            Matrix stateValue = entry.getKey();
            Matrix stateError = entry.getValue();
            optimizer.optimize(stateValue, stateError.divide(stateTransitionValueMap.size()));
        }

        stateTransitionValueMap.clear();

        // Allows other threads to get execution time.
        try {
            Thread.sleep(0, 1);
        }
        catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(exception);
        }

        updateComplete();
        reset();
    }

    /**
     * Updates parameters to this tabular function estimator from another tabular function estimator.
     *
     * @param functionEstimator estimator function used to update this function.
     * @param fullUpdate if true full update is done.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void append(FunctionEstimator functionEstimator, boolean fullUpdate) throws AgentException, MatrixException, NeuralNetworkException, IOException, DynamicParamException, ClassNotFoundException {
        super.append();
        setStateValues(((TabularFunctionEstimator) functionEstimator).getStateValues());
        finalizeAppend();
    }

    /**
     * Appends parameters to this function estimator from another function estimator.
     *
     * @param functionEstimator function estimator used to update current function estimator.
     * @param tau tau which controls contribution of other function estimator.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void append(FunctionEstimator functionEstimator, double tau) throws MatrixException, AgentException, NeuralNetworkException, IOException, DynamicParamException, ClassNotFoundException {
        super.append();
        setStateValues(((TabularFunctionEstimator) functionEstimator).getStateValues());
        finalizeAppend();
    }

    /**
     * Sets if importance sampling weights are applied.
     *
     * @param applyImportanceSamplingWeights if true importance sampling weights are applied otherwise not.
     */
    public void setEnableImportanceSamplingWeights(boolean applyImportanceSamplingWeights) {
    }

}
