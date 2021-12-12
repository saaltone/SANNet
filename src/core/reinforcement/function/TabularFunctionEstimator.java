/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.function;

import core.optimization.*;
import core.reinforcement.agent.AgentException;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.StateTransition;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;

/**
 * Class that defines tabular state action function estimator.<br>
 * Reference for polynomial learning rate: https://www.jmlr.org/papers/volume5/evendar03a/evendar03a.pdf <br>
 *
 */
public class TabularFunctionEstimator extends AbstractFunctionEstimator {

    /**
     * Parameter name types for TabularFunctionEstimator.
     *     - optimizerName: name of optimizer for TabularFunctionEstimator. Default value "Adam".<br>
     *     - learningRate: learning rate for optimizer. Default value 0.01.<br>
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
     * Optimizer for TabularFunctionEstimator.
     *
     */
    private Optimizer optimizer;

    /**
     * Intermediate map for state transition value pairs for function update.
     *
     */
    private HashMap<StateTransition, Matrix> stateTransitionValueMap = new HashMap<>();

    /**
     * Constructor for TabularFunctionEstimator.
     *
     * @param memory memory reference.
     * @param numberOfStates number of states for TabularFunctionEstimator
     * @param numberOfActions number of actions for TabularFunctionEstimator
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public TabularFunctionEstimator(Memory memory, int numberOfStates, int numberOfActions) throws DynamicParamException {
        super (memory, numberOfStates, numberOfActions, false);
    }

    /**
     * Constructor for TabularFunctionEstimator.
     *
     * @param memory memory reference.
     * @param numberOfStates number of states for TabularFunctionEstimator
     * @param numberOfActions number of actions for TabularFunctionEstimator
     * @param params params for TabularFunctionEstimator
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public TabularFunctionEstimator(Memory memory, int numberOfStates, int numberOfActions, String params) throws DynamicParamException {
        super (memory, numberOfStates, numberOfActions, false, params);
    }

    /**
     * Constructor for TabularFunctionEstimator.
     *
     * @param memory memory reference.
     * @param numberOfStates number of states for TabularFunctionEstimator
     * @param numberOfActions number of actions for TabularFunctionEstimator
     * @param stateValues state values inherited for TabularFunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public TabularFunctionEstimator(Memory memory, int numberOfStates, int numberOfActions, HashMap<Matrix, Matrix> stateValues) throws DynamicParamException {
        super (memory, numberOfStates, numberOfActions, false);
        this.stateValues = stateValues;
    }

    /**
     * Constructor for TabularFunctionEstimator.
     *
     * @param memory memory reference.
     * @param numberOfStates number of states for TabularFunctionEstimator
     * @param numberOfActions number of actions for TabularFunctionEstimator
     * @param stateValues state values inherited for TabularFunctionEstimator.
     * @param params params for TabularFunctionEstimator
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
        optimizer = new Adam("learningRate = 0.01");
    }

    /**
     * Returns parameters used for TabularFunctionEstimator.
     *
     * @return parameters used for TabularFunctionEstimator.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + TabularFunctionEstimator.paramNameTypes;
    }

    /**
     * Sets parameters used for TabularFunctionEstimator.<br>
     * <br>
     * Supported parameters are:<br>
     *     - optimizerName: name of optimizer for TabularFunctionEstimator. Default value "Adam".<br>
     *     - learningRate: learning rate for optimizer. Default value 0.01.<br>
     *
     * @param params parameters used for NNFunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("optimizerName")) {
            String optimizerName = params.getValueAsString("optimizerName");
            double learningRate = 0.01;
            if (params.hasParam("learningRate")) learningRate = params.getValueAsDouble("learningRate");
            optimizer = OptimizerFactory.create(optimizerName, "learningRate = " + learningRate);
        }
    }

    /**
     * Returns reference to function estimator.
     *
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator reference() throws DynamicParamException {
        return new TabularFunctionEstimator(getMemory(), getNumberOfStates(), getNumberOfActions(), getParams());
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
     * Sets state values map for TabularFunctionEstimator.
     *
     * @param stateValues state values map
     */
    private void setStateValues(HashMap<Matrix, Matrix> stateValues) {
        this.stateValues = stateValues;
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
        Matrix stateValue = new DMatrix(numberOfActions, 1, Initialization.RANDOM);
        stateValues.put(state.copy(), stateValue);
        return stateValue;
    }

    /**
     * Returns shallow copy of TabularFunctionEstimator.
     *
     * @return shallow copy of TabularFunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator copy() throws DynamicParamException {
        return new TabularFunctionEstimator(memory, getNumberOfStates(), getNumberOfActions(), stateValues, getParams());
    }

    /**
     * Resets TabularFunctionEstimator.
     *
     */
    public void reset() {
        super.reset();
        stateTransitionValueMap = new HashMap<>();
    }

    /**
     * Reinitializes TabularFunctionEstimator.
     *
     */
    public void reinitialize() {
        this.reset();
    }

    /**
     * Returns (predicts) state value corresponding to a state as stored by TabularFunctionEstimator.
     *
     * @param state state
     * @return state value corresponding to a state
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix predict(Matrix state) throws MatrixException {
        return getStateValue(state);
    }

    /**
     * Stores state transition values pair.
     *
     * @param stateTransition state transition.
     * @param values values.
     */
    public void store(StateTransition stateTransition, Matrix values) {
        stateTransitionValueMap.put(stateTransition, values);
    }

    /**
     * Updates (trains) TabularFunctionEstimator.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void update() throws MatrixException, AgentException, DynamicParamException {
        HashMap<Matrix, Matrix> stateErrors = new HashMap<>();
        for (StateTransition stateTransition : stateTransitionValueMap.keySet()) {
            Matrix stateValue = predict(stateTransition.environmentState.state());
            Matrix error = stateValue.subtract(stateTransitionValueMap.get(stateTransition));
            if (!stateErrors.containsKey(stateValue)) stateErrors.put(stateValue, error);
            else stateErrors.get(stateValue).add(error, stateErrors.get(stateValue));
        }
        for (Matrix stateValue : stateErrors.keySet()) {
            optimizer.optimize(stateValue, stateErrors.get(stateValue).divide(stateTransitionValueMap.size()));
        }

        stateTransitionValueMap = new HashMap<>();

        // Allows other threads to get execution time.
        try {
            Thread.sleep(0, 1);
        }
        catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(exception);
        }

        updateComplete();
    }

    /**
     * Updates parameters to this TabularFunctionEstimator from another TabularFunctionEstimator.
     *
     * @param functionEstimator estimator function used to update this function.
     * @param fullUpdate if true full update is done.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void append(FunctionEstimator functionEstimator, boolean fullUpdate) throws AgentException {
        super.append();
         ((TabularFunctionEstimator) functionEstimator).setStateValues(stateValues);
    }

}
