/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import utils.matrix.Matrix;

import java.util.HashSet;

/**
 * Interface for ExecutablePolicy.
 *
 */
public interface ExecutablePolicy {

    /**
     * Increments policy.
     *
     */
    void increment();

    /**
     * Takes action based on policy.
     *
     * @param stateValueMatrix current state value matrix.
     * @param availableActions available actions in current state
     * @param stateValueOffset state value offset
     * @return action taken.
     */
    int action(Matrix stateValueMatrix, HashSet<Integer> availableActions, int stateValueOffset);

}
