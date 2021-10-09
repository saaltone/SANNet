/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.memory;

import utils.Configurable;
import utils.DynamicParamException;

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
     * Returns size of memory.
     *
     * @return size of memory.
     */
    int size();

    /**
     * Adds state transition into memory.
     *
     * @param stateTransition sample to be stored.
     */
    void add(StateTransition stateTransition);

    /**
     * Updates state transition in memory with new error value.
     *
     * @param stateTransition state transition to be updated.
     */
    void update(StateTransition stateTransition);

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
     * Samples defined number of random state transitions.
     *
     * @return retrieved state transitions.
     */
    TreeSet<StateTransition> getRandomStateTransitions();

    /**
     * Returns true if memory contains importance sampling weights and they are to be applied otherwise returns false.
     *
     * @return true if memory contains importance sampling weights and they are to be applied otherwise returns false.
     */
    boolean applyImportanceSamplingWeights();

}
