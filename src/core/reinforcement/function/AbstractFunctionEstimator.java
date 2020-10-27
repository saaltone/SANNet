/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.function;

import core.reinforcement.Agent;
import core.reinforcement.AgentException;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.StateTransition;

import java.io.Serializable;
import java.util.HashSet;
import java.util.TreeSet;

/**
 * Class that implements AbstractFunctionEstimator containing memory management operations and agent handling.
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
     * Active agents for ongoing function estimator update cycle.
     *
     */
    private final HashSet<Agent> activeAgents = new HashSet<>();

    /**
     * Memory instance used by function estimator.
     *
     */
    protected final Memory memory;

    /**
     * If true function is combined state action value function.
     *
     */
    protected boolean isStateActionValueFunction = false;

    /**
     * Number of actions for function estimator.
     *
     */
    protected final int numberOfActions;

    /**
     * Constructor for AbstractFunctionEstimator.
     *
     * @param memory memory reference.
     * @param numberOfActions number of actions.
     */
    public AbstractFunctionEstimator(Memory memory, int numberOfActions) {
        this.memory = memory;
        this.numberOfActions = numberOfActions;
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
     * Resets FunctionEstimator.
     *
     */
    public void reset() {
        memory.reset();
    }

    /**
     * Samples memory of AbstractFunctionEstimator.
     *
     */
    public void sample() {
        memory.sample();
    }

    /**
     * Returns true if sample set is empty after sampling.
     *
     * @return true if sample set is empty after sampling.
     */
    public boolean sampledSetEmpty() {
        return memory.sampledSize() == 0;
    }

    /**
     * Returns sampled state transitions.
     *
     * @return sampled state transitions.
     */
    public TreeSet<StateTransition> getSampledStateTransitions() {
        return memory.getStateTransitions();
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
     * Updated state transition in memory of AbstractFunctionEstimator.
     *
     * @param stateTransition state transition
     */
    public void update(StateTransition stateTransition) {
        memory.update(stateTransition);
    }

    /**
     * Updated state transitions in memory of AbstractFunctionEstimator.
     *
     * @param stateTransitions state transitions
     */
    public void update(TreeSet<StateTransition> stateTransitions) {
        memory.update(stateTransitions);
    }

    /**
     * If true value function is combined state action value function.
     *
     * @return true if value function is combined state action value function.
     */
    public boolean isStateActionValue() {
        return isStateActionValueFunction;
    }

    /**
     * Stores state transition values pair.
     *
     * @param agent deep agent.
     * @throws AgentException throws exception if agent is not registered for ongoing update cycle.
     */
    public void store(Agent agent) throws AgentException {
        if (!registeredAgents.contains(agent)) throw new AgentException("Agent is not registered for function estimator.");
        activeAgents.add(agent);
    }

    /**
     * Updates (trains) estimator and checks if all agents are ready for update.
     *
     * @param agent agent.
     * @throws AgentException throws exception if agent is not registered for ongoing update cycle.
     * @return returns true if all agents are ready for update.
     */
    public boolean updateAndCheck(Agent agent) throws AgentException {
        if (!registeredAgents.contains(agent)) throw new AgentException("Agent is not registered for function estimator.");
        activeAgents.remove(agent);
        return activeAgents.isEmpty();
    }

    /**
     * Appends parameters to this AbstractFunctionEstimator from another AbstractFunctionEstimator.
     *
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void append() throws AgentException {
        if (!activeAgents.isEmpty()) throw new AgentException("Update cycle is ongoing.");
    }

}
