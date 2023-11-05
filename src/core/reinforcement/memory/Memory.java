/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.memory;

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
     * Adds state into memory.
     *
     * @param state sample to be stored.
     */
    void add(State state);

    /**
     * Updates states in memory with new error values.
     *
     * @param states states.
     */
    void update(TreeSet<State> states);

    /**
     * Resets memory.
     *
     */
    void reset();

    /**
     * Samples memory.
     *
     */
    void sample();

    /**
     * Samples defined number of states from memory.
     *
     * @return retrieved states.
     */
    TreeSet<State> getSampledStates();

    /**
     * Returns true if memory contains importance sampling weights, and they are to be applied otherwise returns false.
     *
     * @return true if memory contains importance sampling weights, and they are to be applied otherwise returns false.
     */
    boolean applyImportanceSamplingWeights();

}
