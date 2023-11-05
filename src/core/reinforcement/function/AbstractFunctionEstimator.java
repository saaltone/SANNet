/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.function;

import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.memory.Memory;
import core.reinforcement.agent.State;
import utils.configurable.Configurable;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;

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
     * Constructor for abstract function estimator.
     *
     * @param memory                     memory reference.
     * @param numberOfStates             number of states.
     * @param numberOfActions            number of actions.
     * @param isStateActionValueFunction if true function is combined state action value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractFunctionEstimator(Memory memory, int numberOfStates, int numberOfActions, boolean isStateActionValueFunction) throws DynamicParamException {
        this(memory, numberOfStates, numberOfActions, isStateActionValueFunction, null);
    }

    /**
     * Constructor for abstract function estimator.
     *
     * @param memory                     memory reference.
     * @param numberOfStates             number of states.
     * @param numberOfActions            number of actions.
     * @param isStateActionValueFunction if true function is combined state action value function.
     * @param params                     parameters for function
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
        return memory.getParamDefs();
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
     * Returns sampled states.
     *
     * @return sampled states.
     */
    public TreeSet<State> getSampledStates() {
        return memory.getSampledStates();
    }

    /**
     * Adds new state into memory of abstract function estimator.
     *
     * @param state state
     */
    public void add(State state) {
        memory.add(state);
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
     * Updates states in memory of abstract function estimator.
     *
     * @param states states
     */
    public void update(TreeSet<State> states) {
        memory.update(states);
    }

    /**
     * Aborts function estimator update.
     *
     */
    public void abortUpdate() {
        completedAgents.clear();
    }

    /**
     * Completes abstract function estimator update.
     *
     */
    protected void updateComplete() {
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
     * Returns max value of state given available actions.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return max value of state.
     */
    public double max(Matrix stateValues, HashSet<Integer> availableActions) {
        return stateValues.getValue(argmax(stateValues, availableActions), 0, 0);
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
            double actionValue = stateValues.getValue(action, 0, 0);
            if (maxValue == Double.NEGATIVE_INFINITY || maxValue < actionValue) {
                maxValue = actionValue;
                maxAction = action;
            }
        }
        return maxAction;
    }

}
