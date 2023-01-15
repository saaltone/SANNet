/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.memory;

import core.reinforcement.agent.StateTransition;
import utils.configurable.Configurable;
import utils.configurable.DynamicParamException;

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
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    Memory reference() throws DynamicParamException;

    /**
     * Adds state transition into memory.
     *
     * @param stateTransition sample to be stored.
     */
    void add(StateTransition stateTransition);

    /**
     * Updates state transitions in memory with new error values.
     *
     * @param stateTransitions state transitions.
     */
    void update(TreeSet<StateTransition> stateTransitions);

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
     * Samples defined number of state transitions from memory.
     *
     * @return retrieved state transitions.
     */
    TreeSet<StateTransition> getSampledStateTransitions();

    /**
     * Returns true if memory contains importance sampling weights, and they are to be applied otherwise returns false.
     *
     * @return true if memory contains importance sampling weights, and they are to be applied otherwise returns false.
     */
    boolean applyImportanceSamplingWeights();

}
