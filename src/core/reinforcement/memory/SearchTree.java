/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.memory;

import core.reinforcement.agent.StateTransition;

/**
 * Interface for search tree.
 *
 */
public interface SearchTree {

    /**
     * Current size of search tree i.e. number of state transitions stored.
     *
     * @return size of search tree.
     */
    int size();

    /**
     * Returns total priority of search tree.
     *
     * @return total priority of search tree.
     */
    double getTotalPriority();

    /**
     * Adds state transition in search tree at the location of current node and shifts current node one forward.
     * Updates total priority of search tree according to priority of added state transition.
     *
     * @param stateTransition state transition to be added.
     */
    void add(StateTransition stateTransition);

    /**
     * Updates priority of state transition and entire search tree.
     *
     * @param stateTransition state transition to be updated.
     */
    void update(StateTransition stateTransition);

    /**
     * Returns state transition by priority sum.
     *
     * @param prioritySum priority sum.
     * @return state transition according to priority sum.
     */
    StateTransition getStateTransition(double prioritySum);

    /**
     * Returns random state transition.
     *
     * @return random state transition.
     */
    StateTransition getRandomStateTransition();

}
