/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.NeuralNetworkException;
import core.reinforcement.Agent;
import core.reinforcement.AgentException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import utils.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashSet;
import java.util.TreeSet;

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
     * If true function estimator is state action value function.
     *
     */
    private final boolean isStateActionValueFunction;

    /**
     * Constructor for AbstractValueFunctionEstimator
     *
     * @param numberOfActions number of actions for AbstractValueFunctionEstimator.
     * @param functionEstimator reference to FunctionEstimator.
     */
    public AbstractValueFunctionEstimator(int numberOfActions, FunctionEstimator functionEstimator) {
        super(numberOfActions);
        this.functionEstimator = functionEstimator;
        isStateActionValueFunction = functionEstimator.isStateActionValue();
    }

    /**
     * Constructor for AbstractValueFunctionEstimator
     *
     * @param numberOfActions number of actions for AbstractValueFunctionEstimator.
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractValueFunctionEstimator(int numberOfActions, FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super(numberOfActions, params);
        this.functionEstimator = functionEstimator;
        isStateActionValueFunction = functionEstimator.isStateActionValue();
    }

    /**
     * Starts FunctionEstimator
     *
     * @throws NeuralNetworkException throws exception if starting of value FunctionEstimator fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public void start() throws NeuralNetworkException, MatrixException {
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
     * Returns action with potential state action value offset.
     *
     * @param action action.
     * @return updated action.
     */
    protected int getAction(int action) {
        return (isStateActionValueFunction ? 1 : 0) + action;
    }

    /**
     * Returns values for state.
     *
     * @param stateTransition state transition.
     * @return values for state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix getValues(StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
        return functionEstimator.predict(stateTransition.environmentState.state);
    }

    /**
     * Returns value for state.
     *
     * @param stateTransition state transition.
     * @return value for state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected double getValue(StateTransition stateTransition) throws NeuralNetworkException, MatrixException {
        return getValues(stateTransition).getValue(getAction(stateTransition.action), 0);
    }

    /**
     * Updates baseline value for state transitions.
     *
     * @param stateTransitions state transitions.
     */
    protected void updateBaseline(TreeSet<StateTransition> stateTransitions) {
    }

    /**
     * Updates FunctionEstimator.
     *
     * @param agent agent.
     * @param stateTransitions state transitions used to update FunctionEstimator.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    public void updateFunctionEstimator(Agent agent, TreeSet<StateTransition> stateTransitions) throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException {
        functionEstimator.update(stateTransitions);

        if (!isStateActionValueFunction) {
            for (StateTransition stateTransition : stateTransitions) {
                Matrix targetValues = getValues(stateTransition).copy();
                targetValues.setValue(getAction(stateTransition.action), 0, stateTransition.tdTarget);
                functionEstimator.store(agent, stateTransition, targetValues);
            }
            getFunctionEstimator().update(agent);
        }
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
     * Returns max value of state given available actions.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return max value of state.
     */
    protected double max(Matrix stateValues, HashSet<Integer> availableActions) {
        return stateValues.getValue(getAction(argmax(stateValues, availableActions)), 0);
    }

    /**
     * Returns action with maximum state value given available actions.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return action with maximum state value.
     */
    protected int argmax(Matrix stateValues, HashSet<Integer> availableActions) {
        int maxAction = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int action : availableActions) {
            double actionValue = stateValues.getValue(getAction(action), 0);
            if (maxValue == Double.NEGATIVE_INFINITY || actionValue < maxValue) {
                maxValue = actionValue;
                maxAction = action;
            }
        }
        return maxAction;
    }

}
