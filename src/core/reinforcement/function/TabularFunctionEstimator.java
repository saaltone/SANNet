/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.function;

import core.optimization.*;
import core.reinforcement.AgentException;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.StateTransition;
import utils.DynamicParamException;
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
     * Hash map to store state values pairs.
     *
     */
    private HashMap<Matrix, Matrix> stateValues = new HashMap<>();

    /**
     * Optimizer for TabularFunctionEstimator.
     *
     */
    private Optimizer optimizer = new RAdam();

    /**
     * Intermediate map for state transition value pairs for function update.
     *
     */
    private HashMap<StateTransition, Matrix> stateTransitionValueMap = new HashMap<>();

    /**
     * Constructor for TabularFunctionEstimator.
     *
     * @param memory memory reference.
     * @param numberOfActions number of actions for TabularFunctionEstimator
     */
    public TabularFunctionEstimator(Memory memory, int numberOfActions) {
        super (memory, numberOfActions, false);
    }

    /**
     * Constructor for TabularFunctionEstimator.
     *
     * @param memory memory reference.
     * @param numberOfActions number of actions for TabularFunctionEstimator
     * @param optimizer optimizer
     */
    public TabularFunctionEstimator(Memory memory, int numberOfActions, Optimizer optimizer) {
        super (memory, numberOfActions, false);
        this.optimizer = optimizer;
    }

    /**
     * Constructor for TabularFunctionEstimator.
     *
     * @param memory memory reference.
     * @param numberOfActions number of actions for TabularFunctionEstimator
     * @param stateValues state values inherited for TabularFunctionEstimator.
     */
    public TabularFunctionEstimator(Memory memory, int numberOfActions, HashMap<Matrix, Matrix> stateValues) {
        this(memory, numberOfActions);
        this.stateValues = stateValues;
    }

    /**
     * Constructor for TabularFunctionEstimator.
     *
     * @param memory memory reference.
     * @param numberOfActions number of actions for TabularFunctionEstimator
     * @param stateValues state values inherited for TabularFunctionEstimator.
     * @param optimizer optimizer
     */
    public TabularFunctionEstimator(Memory memory, int numberOfActions, HashMap<Matrix, Matrix> stateValues, Optimizer optimizer) {
        this(memory, numberOfActions, stateValues);
        this.optimizer = optimizer;
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
     */
    public FunctionEstimator copy() {
        return new TabularFunctionEstimator(memory, getNumberOfActions(), stateValues, optimizer);
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
            Matrix stateValue = predict(stateTransition.environmentState.state);
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
