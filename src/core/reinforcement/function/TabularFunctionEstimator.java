/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.function;

import core.reinforcement.Agent;
import core.reinforcement.AgentException;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.StateTransition;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;

/**
 * Class that defines tabular state action function estimator.<br>
 * Reference for polynomial learning rate: https://www.jmlr.org/papers/volume5/evendar03a/evendar03a.pdf.<br>
 *
 */
public class TabularFunctionEstimator extends AbstractFunctionEstimator {

    /**
     * Hash map to store state values pairs.
     *
     */
    private HashMap<Matrix, Matrix> stateValues = new HashMap<>();

    /**
     * Constant learning rate for training.
     *
     */
    private double learningRate = 0.1;

    /**
     * If true uses polynomial learning rate otherwise constant learning rate.
     *
     */
    private boolean usePolynomialLearningRate = true;

    /**
     * Omega value adjusting polynomial learning rate.
     *
     */
    private double omega = 0.1;

    /**
     * Current learning time step.
     *
     */
    private int timeStep = 0;

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
        super (memory, numberOfActions);
    }

    /**
     * Constructor for TabularFunctionEstimator.
     *
     * @param memory memory reference.
     * @param numberOfActions number of actions for TabularFunctionEstimator
     * @param params parameters for function
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public TabularFunctionEstimator(Memory memory, int numberOfActions, String params) throws DynamicParamException {
        this(memory, numberOfActions);
        setParams(new DynamicParam(params, getParamDefs()));
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
     * Returns parameters used for TabularFunctionEstimator.
     *
     * @return parameters used for TabularFunctionEstimator.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("learningRate", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("usePolynomialLearningRate", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("omega", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for TabularFunctionEstimator.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for updates. Default value 0.1.<br>
     *     - usePolynomialLearningRate: if true polynomial learning rate is used. Default value false.<br>
     *     - omega: value adjusting polynomial learning rate. Default value 0.65.<br>
     *
     * @param params parameters used for TabularFunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("learningRate")) learningRate = params.getValueAsDouble("learningRate");
        if (params.hasParam("usePolynomialLearningRate")) usePolynomialLearningRate = params.getValueAsBoolean("usePolynomialLearningRate");
        if (params.hasParam("omega")) omega = params.getValueAsDouble("omega");
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
        Matrix stateValue = new DMatrix(numberOfActions, 1);
        stateValues.put(state.copy(), stateValue);
        return stateValue;
    }

    /**
     * Returns shallow copy of TabularFunctionEstimator.
     *
     * @return shallow copy of TabularFunctionEstimator.
     */
    public FunctionEstimator copy() {
        return new TabularFunctionEstimator(memory, getNumberOfActions(), stateValues);
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
     * @param agent agent.
     * @param stateTransition state transition.
     * @param values values.
     * @throws AgentException throws exception if agent tries to store values outside ongoing update cycle or one of agent are not ready to initiate update cycle.
     */
    public void store(Agent agent, StateTransition stateTransition, Matrix values) throws AgentException {
        super.store(agent);
        stateTransitionValueMap.put(stateTransition, values);
    }

    /**
     * Updates (trains) TabularFunctionEstimator.
     *
     * @param agent agent.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws AgentException throws exception if agent is not registered for ongoing update cycle.
     */
    public void update(Agent agent) throws MatrixException, AgentException {
        if (!super.updateAndCheck(agent)) return;

        for (StateTransition stateTransition : stateTransitionValueMap.keySet()) {
            // currentStateValue: Q(s,a) stored by TabularFunctionEstimator
            // targetStateValue: reward + gamma * targetValue per updated TD target
            // Q(s,a) = Q(s,a) + learningRate * (reward + gamma * targetValue - Q(s,a))
            Matrix currentStateValue = predict(stateTransition.environmentState.state);
            Matrix targetStateValue = stateTransitionValueMap.get(stateTransition);
            currentStateValue.add(targetStateValue.subtract(currentStateValue).multiply(getLearningRate()), currentStateValue);
        }

        stateTransitionValueMap = new HashMap<>();

        // Allows other threads to get execution time.
        try {
            Thread.sleep(1);
        } catch (InterruptedException e) {}
    }

    /**
     * Returns current learning rate
     *
     * @return current learning rate.
     */
    private double getLearningRate() {
        return usePolynomialLearningRate ? 1 / Math.pow(1 + ++timeStep, omega) : learningRate;
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
