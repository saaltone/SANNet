/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement;

import utils.matrix.Matrix;

import java.io.Serializable;
import java.util.HashSet;

/**
 * Record that defines state of environment.<br>
 * <br>
 * @param episodeID episode ID.
 * @param timeStep time step.
 * @param state state of environment.
 * @param availableActions actions available at state.
 *
 */
public record EnvironmentState(int episodeID, int timeStep, Matrix state, HashSet<Integer> availableActions) implements Serializable, Comparable<EnvironmentState> {

    /**
     * Compares this environment state to other environment state.<br>
     * If other environment state is precedent to this environment state returns 1.<br>
     * If other environment state succeeds this environment state returns -1.<br>
     * Otherwise returns 0.<br>
     *
     * @param otherEnvironmentState other environment state
     * @return return value of comparison.
     */
    public int compareTo(EnvironmentState otherEnvironmentState) {
        return episodeID > otherEnvironmentState.episodeID ? 1 : episodeID < otherEnvironmentState.episodeID ? -1 : Integer.compare(timeStep, otherEnvironmentState.timeStep);
    }

    /**
     * Prints environment state.
     *
     */
    public void print() {
        System.out.println("Episode ID: " + episodeID + " Time step: " + timeStep);
        state.print();
        System.out.println(availableActions);
    }

}
