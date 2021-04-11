/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.function;

import core.NeuralNetworkException;
import core.reinforcement.Agent;
import core.reinforcement.AgentException;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.StateTransition;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeSet;

/**
 * Class that implements AbstractFunctionEstimator containing memory management operations and agent handling.<br>
 *
 */
public abstract class AbstractFunctionEstimator implements FunctionEstimator, Serializable {

    private static final long serialVersionUID = -557430597852291426L;

    /**
     * Agents registered for function estimator.
     *
     */
    private final HashSet<Agent> registeredAgents = new HashSet<>();

    /**
     * Agents ready for function estimator update.
     *
     */
    private final HashSet<Agent> completedAgents = new HashSet<>();

    /**
     * Memory instance used by function estimator.
     *
     */
    protected final Memory memory;

    /**
     * If true function is combined state action value function.
     *
     */
    protected final boolean isStateActionValueFunction;

    /**
     * Number of actions for function estimator.
     *
     */
    protected final int numberOfActions;

    /**
     * Target function estimator.
     *
     */
    private FunctionEstimator targetFunctionEstimator = null;

    /**
     * Update cycle (in episodes) for target FunctionEstimator. If update cycle is zero then smooth parameter updates are applied with update rate tau.
     *
     */
    private int targetFunctionUpdateCycle = 0;

    /**
     * Update count for update cycle.
     *
     */
    private transient int updateCount = 0;

    /**
     * Constructor for AbstractFunctionEstimator.
     *
     * @param memory memory reference.
     * @param numberOfActions number of actions.
     * @param isStateActionValueFunction if true function is combined state action value function.
     */
    public AbstractFunctionEstimator(Memory memory, int numberOfActions, boolean isStateActionValueFunction) {
        this.memory = memory;
        this.numberOfActions = numberOfActions;
        this.isStateActionValueFunction = isStateActionValueFunction;
    }

    /**
     * Constructor for AbstractFunctionEstimator.
     *
     * @param memory memory reference.
     * @param numberOfActions number of actions.
     * @param isStateActionValueFunction if true function is combined state action value function.
     * @param params parameters for function
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractFunctionEstimator(Memory memory, int numberOfActions, boolean isStateActionValueFunction, String params) throws DynamicParamException {
        this(memory, numberOfActions, isStateActionValueFunction);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for AbstractFunctionEstimator.
     *
     * @return parameters used for AbstractFunctionEstimator.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(memory.getParamDefs());
        paramDefs.put("targetFunctionUpdateCycle", DynamicParam.ParamType.INT);
        return paramDefs;
    }

    /**
     * Sets parameters used for AbstractFunctionEstimator.<br>
     * <br>
     * Supported parameters are:<br>
     *     - targetFunctionUpdateCycle; target function update cycle. Default value 0 (smooth update).<br>
     *
     * @param params parameters used for AbstractFunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        memory.setParams(params);
        if (params.hasParam("targetFunctionUpdateCycle")) targetFunctionUpdateCycle = params.getValueAsInteger("targetFunctionUpdateCycle");
    }

    /**
     * Starts function estimator.
     *
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException {
        if (targetFunctionEstimator != null) targetFunctionEstimator.start();
    }

    /**
     * Stops function estimator.
     *
     */
    public void stop() {
        if (targetFunctionEstimator != null) targetFunctionEstimator.stop();
    }

    /**
     * Returns number of actions for AbstractFunctionEstimator.
     *
     * @return number of actions for AbstractFunctionEstimator.
     */
    public int getNumberOfActions() {
        return numberOfActions;
    }

    /**
     * Registers agent for AbstractFunctionEstimator.
     *
     * @param agent agent.
     */
    public void registerAgent(Agent agent) {
        registeredAgents.add(agent);
    }

    /**
     * Returns memory of function estimator.
     *
     * @return memory of function estimator.
     */
    public Memory getMemory() {
        return memory;
    }

    /**
     * Resets FunctionEstimator.
     *
     */
    public void reset() {
        memory.reset();
    }

    /**
     * Reinitializes FunctionEstimator.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void reinitialize() throws NeuralNetworkException, MatrixException {
    }

    /**
     * Samples memory of AbstractFunctionEstimator.
     *
     */
    public void sample() {
        memory.sample();
    }

    /**
     * Returns sampled state transitions.
     *
     * @return sampled state transitions.
     */
    public TreeSet<StateTransition> getSampledStateTransitions() {
        return memory.getSampledStateTransitions();
    }

    /**
     * Adds new state transition into memory of AbstractFunctionEstimator.
     *
     * @param stateTransition state transition
     */
    public void add(StateTransition stateTransition) {
        memory.add(stateTransition);
    }

    /**
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @throws AgentException throws exception if agent is not registered for function estimator.
     * @return true if all registered agents are ready to update.
     */
    public boolean readyToUpdate(Agent agent) throws AgentException {
        if (!registeredAgents.contains(agent)) throw new AgentException("Agent is not registered for function estimator.");
        completedAgents.add(agent);
        return completedAgents.containsAll(registeredAgents);
    }

    /**
     * Updates state transition in memory of AbstractFunctionEstimator.
     *
     * @param stateTransition state transition
     */
    public void update(StateTransition stateTransition) {
        memory.update(stateTransition);
    }

    /**
     * Updates state transitions in memory of AbstractFunctionEstimator.
     *
     * @param stateTransitions state transitions
     */
    public void update(TreeSet<StateTransition> stateTransitions) {
        memory.update(stateTransitions);
    }

    /**
     * Completes AbstractFunctionEstimator update.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    protected void updateComplete() throws AgentException, MatrixException {
        updateTargetFunctionEstimator();
        completedAgents.clear();
    }

    /**
     * If true value function is combined state action value function.
     *
     * @return true if value function is combined state action value function.
     */
    public boolean isStateActionValueFunction() {
        return isStateActionValueFunction;
    }

    /**
     * Appends parameters to this AbstractFunctionEstimator from another AbstractFunctionEstimator.
     *
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void append() throws AgentException {
        if (!completedAgents.isEmpty()) throw new AgentException("Update cycle is ongoing.");
    }

    /**
     * Returns max value of state.
     *
     * @param stateValues state values.
     * @return max value of state.
     */
    public double max(Matrix stateValues) {
        return stateValues.getValue(getAction(argmax(stateValues)), 0);
    }

    /**
     * Returns max value of state given available actions.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return max value of state.
     */
    public double max(Matrix stateValues, HashSet<Integer> availableActions) {
        return stateValues.getValue(getAction(argmax(stateValues, availableActions)), 0);
    }

    /**
     * Returns action with maximum state value.
     *
     * @param stateValues state values.
     * @return action with maximum state value.
     */
    public int argmax(Matrix stateValues) {
        int maxAction = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int action = 0; action < getNumberOfActions(); action++) {
            double actionValue = stateValues.getValue(getAction(action), 0);
            if (maxValue == Double.NEGATIVE_INFINITY || maxValue < actionValue) {
                maxValue = actionValue;
                maxAction = action;
            }
        }
        return maxAction;
    }

    /**
     * Returns action with maximum state value given available actions.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return action with maximum state value.
     */
    public int argmax(Matrix stateValues, HashSet<Integer> availableActions) {
        int maxAction = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int action : availableActions) {
            double actionValue = stateValues.getValue(getAction(action), 0);
            if (maxValue == Double.NEGATIVE_INFINITY || maxValue < actionValue) {
                maxValue = actionValue;
                maxAction = action;
            }
        }
        return maxAction;
    }

    /**
     * Returns action with potential state action value offset.
     *
     * @param action action.
     * @return updated action.
     */
    private int getAction(int action) {
        return (isStateActionValueFunction() ? 1 : 0) + action;
    }

    /**
     * Sets target function estimator.
     *
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void setTargetFunctionEstimator() throws ClassNotFoundException, MatrixException, DynamicParamException, IOException {
        targetFunctionEstimator = copy();
    }

    /**
     * Returns target function estimator.
     *
     * @return target function estimator.
     */
    public FunctionEstimator getTargetFunctionEstimator() {
        return targetFunctionEstimator;
    }

    /**
     * Updates target function estimator.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    private void updateTargetFunctionEstimator() throws AgentException, MatrixException {
        if (targetFunctionEstimator == null) return;
        if (targetFunctionUpdateCycle == 0) targetFunctionEstimator.append(this, false);
        else {
            if (++updateCount >= targetFunctionUpdateCycle) {
                targetFunctionEstimator.append(this, true);
                updateCount = 0;
            }
        }
    }

}
