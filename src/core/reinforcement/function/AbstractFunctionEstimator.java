/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.function;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.StateTransition;
import utils.configurable.Configurable;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.io.Serial;
import java.io.Serializable;
import java.util.HashSet;
import java.util.TreeSet;

/**
 * Implements abstract function estimator containing memory management operations and agent handling.<br>
 *
 */
public abstract class AbstractFunctionEstimator implements Configurable, FunctionEstimator, Serializable {

    @Serial
    private static final long serialVersionUID = -557430597852291426L;

    /**
     * Parameter name types for abstract function estimator.
     *     - targetFunctionUpdateCycle; target function update cycle. Default value 0 (smooth update).<br>
     *
     */
    private final static String paramNameTypes = "(targetFunctionUpdateCycle:INT)";

    /**
     * Parameters for function estimator.
     *
     */
    private final String params;

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
     * Number of states for function estimator.
     *
     */
    protected final int numberOfStates;

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
     * Update cycle (in episodes) for target function estimator. If update cycle is zero then smooth parameter updates are applied with update rate tau.
     *
     */
    private int targetFunctionUpdateCycle;

    /**
     * Update count for target function update cycle.
     *
     */
    private transient int updateCount = 0;

    /**
     * Constructor for abstract function estimator.
     *
     * @param memory memory reference.
     * @param numberOfStates number of states.
     * @param numberOfActions number of actions.
     * @param isStateActionValueFunction if true function is combined state action value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractFunctionEstimator(Memory memory, int numberOfStates, int numberOfActions, boolean isStateActionValueFunction) throws DynamicParamException {
        initializeDefaultParams();
        this.memory = memory;
        this.numberOfStates = numberOfStates;
        this.numberOfActions = numberOfActions;
        this.isStateActionValueFunction = isStateActionValueFunction;
        this.params = null;
    }

    /**
     * Constructor for abstract function estimator.
     *
     * @param memory memory reference.
     * @param numberOfStates number of states.
     * @param numberOfActions number of actions.
     * @param isStateActionValueFunction if true function is combined state action value function.
     * @param params parameters for function
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractFunctionEstimator(Memory memory, int numberOfStates, int numberOfActions, boolean isStateActionValueFunction, String params) throws DynamicParamException {
        initializeDefaultParams();
        this.memory = memory;
        this.numberOfStates = numberOfStates;
        this.numberOfActions = numberOfActions;
        this.isStateActionValueFunction = isStateActionValueFunction;
        this.params = params;
        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() throws DynamicParamException {
        targetFunctionUpdateCycle = 0;
    }

    /**
     * Returns parameters of function estimator.
     *
     * @return parameters for function estimator.
     */
    protected String getParams() {
        return params;
    }

    /**
     * Returns parameters used for abstract function estimator.
     *
     * @return parameters used for abstract function estimator.
     */
    public String getParamDefs() {
        return AbstractFunctionEstimator.paramNameTypes + ", " + memory.getParamDefs();
    }

    /**
     * Sets parameters used for abstract function estimator.<br>
     * <br>
     * Supported parameters are:<br>
     *     - targetFunctionUpdateCycle; target function update cycle. Default value 0 (smooth update).<br>
     *
     * @param params parameters used for abstract function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        memory.setParams(params);
        if (params.hasParam("targetFunctionUpdateCycle")) targetFunctionUpdateCycle = params.getValueAsInteger("targetFunctionUpdateCycle");
    }

    /**
     * Returns reference to memory of function estimator.
     *
     * @return reference to memory of function estimator.
     */
    public Memory getMemory() {
        return memory;
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
     * Returns number of states for abstract function estimator.
     *
     * @return number of states for abstract function estimator.
     */
    public int getNumberOfStates() {
        return numberOfStates;
    }

    /**
     * Returns number of actions for abstract function estimator.
     *
     * @return number of actions for abstract function estimator.
     */
    public int getNumberOfActions() {
        return numberOfActions;
    }

    /**
     * Registers agent for abstract function estimator.
     *
     * @param agent agent.
     */
    public void registerAgent(Agent agent) {
        registeredAgents.add(agent);
    }

    /**
     * Resets function estimator.
     *
     */
    public void reset() {
        memory.reset();
    }

    /**
     * Samples memory of abstract function estimator.
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
     * Adds new state transition into memory of abstract function estimator.
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
     * Updates state transitions in memory of abstract function estimator.
     *
     * @param stateTransitions state transitions
     */
    public void update(TreeSet<StateTransition> stateTransitions) {
        memory.update(stateTransitions);
    }

    /**
     * Completes abstract function estimator update.
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
     */
    public void setTargetFunctionEstimator() throws ClassNotFoundException, DynamicParamException, IOException {
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

    /**
     * Appends parameters to this abstract function estimator from another abstract function estimator.
     *
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void append() throws AgentException {
        if (!completedAgents.isEmpty()) throw new AgentException("Update cycle is ongoing.");
    }

    /**
     * Returns min value of state.
     *
     * @param stateValues state values.
     * @return max value of state.
     */
    public double min(Matrix stateValues) {
        return stateValues.getValue(getAction(argmin(stateValues)), 0);
    }

    /**
     * Returns min value of state given available actions.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return min value of state.
     */
    public double min(Matrix stateValues, HashSet<Integer> availableActions) {
        return stateValues.getValue(getAction(argmin(stateValues, availableActions)), 0);
    }

    /**
     * Returns action with minimum state value.
     *
     * @param stateValues state values.
     * @return action with minimum state value.
     */
    public int argmin(Matrix stateValues) {
        int minAction = -1;
        double minValue = Double.POSITIVE_INFINITY;
        for (int action = 0; action < getNumberOfActions(); action++) {
            double actionValue = stateValues.getValue(getAction(action), 0);
            if (minValue == Double.POSITIVE_INFINITY || minValue > actionValue) {
                minValue = actionValue;
                minAction = action;
            }
        }
        return minAction;
    }

    /**
     * Returns action with minimum state value given available actions.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return action with minimum state value.
     */
    public int argmin(Matrix stateValues, HashSet<Integer> availableActions) {
        int minAction = -1;
        double minValue = Double.POSITIVE_INFINITY;
        for (int action : availableActions) {
            double actionValue = stateValues.getValue(getAction(action), 0);
            if (minValue == Double.POSITIVE_INFINITY || minValue > actionValue) {
                minValue = actionValue;
                minAction = action;
            }
        }
        return minAction;
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

}
