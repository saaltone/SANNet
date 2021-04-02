/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.memory;

import utils.Configurable;

import java.util.TreeSet;

/**
 * Interface for Memory.
 *
 */
public interface Memory extends Configurable {

    /**
     * Returns size of Memory.
     *
     * @return size of Memory.
     */
    int size();

    /**
     * Adds state transition into Memory.
     *
     * @param stateTransition sample to be stored.
     */
    void add(StateTransition stateTransition);

    /**
     * Updates state transition in Memory with new error value.
     *
     * @param stateTransition state transition to be updated.
     */
    void update(StateTransition stateTransition);

    /**
     * Updates state transitions in Memory with new error values.
     *
     * @param stateTransitions state transitions.
     */
    void update(TreeSet<StateTransition> stateTransitions);

    /**
     * Resets Memory.
     *
     */
    void reset();

    /**
     * Samples Memory.
     *
     */
    void sample();

    /**
     * Samples defined number of state transitions from Memory.
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
