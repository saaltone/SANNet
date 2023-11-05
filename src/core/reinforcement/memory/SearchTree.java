/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.memory;

import core.reinforcement.agent.State;

/**
 * Interface for search tree.
 *
 */
public interface SearchTree {

    /**
     * Current size of search tree i.e. number of states stored.
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
     * Adds state in search tree at the location of current node and shifts current node one forward.
     * Updates total priority of search tree according to priority of added state.
     *
     * @param state state to be added.
     */
    void add(State state);

    /**
     * Updates priority of state and entire search tree.
     *
     * @param state state to be updated.
     */
    void update(State state);

    /**
     * Returns state by priority sum.
     *
     * @param prioritySum priority sum.
     * @return state according to priority sum.
     */
    State getState(double prioritySum);

    /**
     * Returns random state.
     *
     * @return random state.
     */
    State getRandomState();

}
