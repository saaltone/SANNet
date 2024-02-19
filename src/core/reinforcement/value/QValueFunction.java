/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.State;
import core.reinforcement.function.FunctionEstimator;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.TreeSet;

/**
 * Implements Q value function (Q value function with function estimator).<br>
 *
 */
public class QValueFunction extends AbstractValueFunctionEstimator {

    /**
     * If true soft Q value function estimator has dual function estimator.
     *
     */
    protected final boolean dualFunctionEstimation;

    /**
     * Reference to second function estimator.
     *
     */
    private FunctionEstimator functionEstimator2;

    /**
     * If true uses target value function estimator otherwise not.
     *
     */
    private final boolean usesTargetValueFunctionEstimator;

    /**
     * Constructor for Q value function.
     *
     * @param functionEstimator                reference to function estimator.
     * @param params                           parameters for value function.
     */
    public QValueFunction(FunctionEstimator functionEstimator, String params) {
        this(functionEstimator, null, false, false, params);
    }

    /**
     * Constructor for Q value function.
     *
     * @param functionEstimator                reference to function estimator.
     * @param functionEstimator2               reference to second value function estimator.
     * @param dualFunctionEstimation           if true soft Q value function estimator has dual function estimator.
     * @param usesTargetValueFunctionEstimator if true uses target value function estimator.
     * @param params                           parameters for value function.
     */
    protected QValueFunction(FunctionEstimator functionEstimator, FunctionEstimator functionEstimator2, boolean dualFunctionEstimation, boolean usesTargetValueFunctionEstimator, String params) {
        super(functionEstimator, params);
        this.functionEstimator2 = functionEstimator2;
        this.dualFunctionEstimation = dualFunctionEstimation;
        this.usesTargetValueFunctionEstimator = usesTargetValueFunctionEstimator;
    }

    /**
     * Returns 2nd function estimator.
     *
     * @return 2nd function estimator.
     */
    protected FunctionEstimator getFunctionEstimator2() {
        return functionEstimator2;
    }

    /**
     * Returns if target value function estimator if used.
     *
     * @return if true value function is using target value function estimator otherwise not.
     */
    protected boolean isUsingTargetValueFunctionEstimator() {
        return usesTargetValueFunctionEstimator;
    }

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @return reference to value function.
     * @throws IOException            throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws MatrixException        throws exception if neural network has less output than actions.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new QValueFunction(sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(), sharedValueFunctionEstimator ? getFunctionEstimator2() : null, dualFunctionEstimation, usesTargetValueFunctionEstimator, getParams());
    }

    /**
     * Starts function estimator
     *
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start(Agent agent) throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException {
        super.start(agent);
        if (functionEstimator2 == null && dualFunctionEstimation) {
            functionEstimator2 = getFunctionEstimator().reference();
            if (getFunctionEstimator2() != null) getFunctionEstimator2().start();
        }
        if (functionEstimator2 != null) getFunctionEstimator2().registerAgent(agent);
    }

    /**
     * Stops function estimator
     *
     * @throws NeuralNetworkException throws exception is neural network is not started.
     */
    public void stop() throws NeuralNetworkException {
        super.stop();
        if (getFunctionEstimator2() != null) getFunctionEstimator2().stop();
    }

    /**
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @throws AgentException throws exception if agent is not registered for function estimator.
     * @return true if all registered agents are ready to update.
     */
    public boolean readyToUpdate(Agent agent) throws AgentException {
        return super.readyToUpdate(agent) && (getFunctionEstimator2() == null || getFunctionEstimator2().readyToUpdate(agent));
    }

    /**
     * Prepares function estimator update.
     *
     * @param sampledStates sampled states.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     */
    public void prepareFunctionEstimatorUpdate(TreeSet<State> sampledStates) throws NeuralNetworkException, MatrixException, DynamicParamException {
        super.prepareFunctionEstimatorUpdate(sampledStates);
        if (getFunctionEstimator2() != null) {
            updateTargetValues(getFunctionEstimator2(), sampledStates);
        }
    }

    /**
     * Finishes function estimator update.
     *
     * @param sampledStates sampled states.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     */
    public void finishFunctionEstimatorUpdate(TreeSet<State> sampledStates) throws NeuralNetworkException, MatrixException, DynamicParamException {
        super.finishFunctionEstimatorUpdate(sampledStates);
        if (getFunctionEstimator2() != null && !isStateActionValueFunction())  getFunctionEstimator2().update();
    }

    /**
     * Returns function index applying potential state action value offset.
     *
     * @param state state.
     * @return function index.
     */
    protected int getValueFunctionIndex(State state) {
        return state.action;
    }

    /**
     * Returns target value based on next state. Uses target action with maximal value as defined by target FunctionEstimator.
     *
     * @param nextState next state.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public double getTargetValue(State nextState) throws NeuralNetworkException, MatrixException, DynamicParamException {
        return getTargetValues(nextState, usesTargetValueFunctionEstimator).getValue(getTargetAction(nextState), 0, 0);
    }

    /**
     * Returns target Q value.
     *
     * @param nextState                        next state.
     * @param usesTargetValueFunctionEstimator if true uses target value function estimator.
     * @return target Q value.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public Matrix getTargetValues(State nextState, boolean usesTargetValueFunctionEstimator) throws MatrixException, NeuralNetworkException, DynamicParamException {
        if (getFunctionEstimator2() != null) {
            Matrix values1 = predictStateActionValues(nextState, getFunctionEstimator(), usesTargetValueFunctionEstimator);
            Matrix values2 = predictStateActionValues(nextState, getFunctionEstimator2(), usesTargetValueFunctionEstimator);
            return values1.min(values2);
        }
        else {
            return predictStateActionValues(nextState, getFunctionEstimator(), usesTargetValueFunctionEstimator);
        }
    }

    /**
     * Returns target action based on next state.
     *
     * @param nextState next state.
     * @return target action based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    private int getTargetAction(State nextState) throws NeuralNetworkException, MatrixException {
        return getFunctionEstimator().argmax(predictStateActionValues(nextState, getFunctionEstimator(), false), nextState.environmentState.availableActions());
    }

    /**
     * Predicts state action values.
     *
     * @param nextState next state.
     * @param functionEstimator function estimator.
     * @param usesTargetValueFunctionEstimator if true uses target value function estimator.
     * @return predicted state action values.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix predictStateActionValues(State nextState, FunctionEstimator functionEstimator, boolean usesTargetValueFunctionEstimator) throws MatrixException, NeuralNetworkException {
        return usesTargetValueFunctionEstimator ? functionEstimator.predictTargetStateActionValues(nextState) : functionEstimator.predictStateActionValues(nextState);
    }

}
