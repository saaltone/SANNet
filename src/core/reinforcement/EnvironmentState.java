/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement;

import utils.matrix.Matrix;

import java.io.Serializable;
import java.util.HashSet;

/**
 * Class that defines current state of environment.
 *
 */
public class EnvironmentState implements Serializable, Comparable<EnvironmentState> {

    /**
     * Episode ID
     *
     */
    public final int episodeID;

    /**
     * Time step of episode.
     *
     */
    public final int timeStep;

    /**
     * State of environment at episode ID and time step.
     *
     */
    public final Matrix state;

    /**
     * Actions available at state.
     *
     */
    public final HashSet<Integer> availableActions;

    /**
     * Constructor for environment state.
     *
     * @param episodeID episode ID.
     * @param timeStep time step.
     * @param state state of environment.
     * @param availableActions actions available at state.
     */
    public EnvironmentState(int episodeID, int timeStep, Matrix state, HashSet<Integer> availableActions) {
        this.episodeID = episodeID;
        this.timeStep = timeStep;
        this.state = state;
        this.availableActions = availableActions;
    }

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
