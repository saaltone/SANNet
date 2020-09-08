/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.memory;

import java.util.TreeSet;

/**
 * Interface for Memory.
 *
 */
public interface Memory {

    /**
     * Returns size of Memory.
     *
     * @return size of Memory.
     */
    int size();

    /**
     * Returns size of sampled set.
     *
     * @return size of sampled set.
     */
    int sampledSize();

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
     * Samples Memory.
     *
     */
    void sample();

    /**
     * Samples defined number of state transitions from Memory.
     *
     * @return retrieved state transitions.
     */
    TreeSet<StateTransition> getStateTransitions();

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
