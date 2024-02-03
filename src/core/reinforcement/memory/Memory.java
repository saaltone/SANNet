/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.memory;

import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.State;
import utils.configurable.Configurable;

import java.util.TreeSet;

/**
 * Interface for memory.<br>
 *
 */
public interface Memory extends Configurable {

    /**
     * Returns reference to memory.
     *
     * @return reference to memory.
     */
    Memory reference();

    /**
     * Registers agent for function estimator.
     *
     * @param agent agent.
     */
    void registerAgent(Agent agent);

    /**
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @return returns true memory is ready for update otherwise returns false.
     * @throws AgentException throws exception if agent is not registered for function estimator.
     */
    boolean readyToUpdate(Agent agent) throws AgentException;

    /**
     * Adds state into memory.
     *
     * @param state sample to be stored.
     */
    void add(State state);

    /**
     * Samples memory.
     *
     * @return sampled states.
     */
    TreeSet<State> sample();

    /**
     * Updates states in memory with new error values.
     */
    void update();

    /**
     * Resets memory.
     *
     */
    void reset();

}
