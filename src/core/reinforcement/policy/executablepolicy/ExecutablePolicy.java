/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import core.reinforcement.memory.StateTransition;
import utils.matrix.Matrix;

import java.util.HashSet;

/**
 * Interface for ExecutablePolicy.
 *
 */
public interface ExecutablePolicy {

    /**
     * Resets policy.
     *
     * @param forceReset forces to trigger reset.
     */
    void reset(boolean forceReset);

    /**
     * Increments policy.
     *
     */
    void increment();

    /**
     * Takes action decided by external agent.
     *
     * @param stateValueMatrix current state value matrix.
     * @param availableActions available actions in current state
     * @param stateValueOffset state value offset
     * @param action action.
     */
    void action(Matrix stateValueMatrix, HashSet<Integer> availableActions, int stateValueOffset, int action);

    /**
     * Takes action based on policy.
     *
     * @param stateValueMatrix current state value matrix.
     * @param availableActions available actions in current state
     * @param stateValueOffset state value offset
     * @param alwaysGreedy if true greedy action is always taken.
     * @return action taken.
     */
    int action(Matrix stateValueMatrix, HashSet<Integer> availableActions, int stateValueOffset, boolean alwaysGreedy);

    /**
     * Records state transition for action execution.
     *
     * @param stateTransition state transition.
     */
    void record(StateTransition stateTransition);

    /**
     * Updates policy.
     *
     */
    void update();
    /**
     * Finishes episode.
     *
     */
    void finish();

}
