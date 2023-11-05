/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.agent;

import java.io.Serial;
import java.io.Serializable;

/**
 * Implements state containing information of state matrix, action and available actions, reward and reference to previous and next states.<br>
 *
 */
public class State implements Serializable, Comparable<State> {

    @Serial
    private static final long serialVersionUID = 3018272924414901045L;

    /**
     * Episode ID.
     *
     */
    private final int episodeID;

    /**
     * Time step.
     *
     */
    private final int timeStep;

    /**
     * Current environment state.
     *
     */
    public final EnvironmentState environmentState;

    /**
     * Action taken to move from current environment state to next state.
     *
     */
    public int action = -1;

    /**
     * Immediate reward after taking specific action in current environment state.
     *
     */
    public double reward;

    /**
     * Previous state .
     *
     */
    public State previousState;

    /**
     * Next state.
     *
     */
    public State nextState;

    /**
     * Priority based on TD error.
     *
     */
    public double priority;

    /**
     * Importance sampling weight.
     *
     */
    public double importanceSamplingWeight;

    /**
     * Policy value.
     *
     */
    public double policyValue;

    /**
     * State value.
     *
     */
    public double stateValue;

    /**
     * TD target value.
     *
     */
    public double tdTarget;

    /**
     * TD error.
     *
     */
    public double tdError;

    /**
     * Advantage.
     *
     */
    public double advantage;

    /**
     * Constructor for state.
     *
     * @param episodeID episode ID
     * @param timeStep time step.
     * @param environmentState current environment state.
     */
    public State(int episodeID, int timeStep, EnvironmentState environmentState) {
        this.episodeID = episodeID;
        this.timeStep = timeStep;
        this.environmentState = environmentState;
    }

    /**
     * Returns episode ID.
     *
     * @return episode ID.
     */
    public int getEpisodeID() {
        return episodeID;
    }

    /**
     * Returns time step.
     *
     * @return time step.
     */
    public int getTimeStep() {
        return timeStep;
    }

    /**
     * Checks if environment state is final in episodic learning.
     *
     * @return returns true if state is final otherwise returns false.
     */
    public boolean isFinalState() {
        return nextState == null;
    }

    /**
     * Removes previous state.
     *
     */
    public void removePreviousState() {
        previousState = null;
    }

    /**
     * Compares this state to other state.<br>
     * If other state is precedent to this state returns 1.<br>
     * If other state succeeds this state returns -1.<br>
     * If above conditions are not met returns 0.<br>
     *
     * @param otherState other state.
     * @return return value of comparison.
     */
    public int compareTo(State otherState) {
        return episodeID > otherState.episodeID ? 1 : episodeID < otherState.episodeID ? -1 : Integer.compare(timeStep, otherState.timeStep);
    }

    /**
     * Prints state.
     *
     */
    public void print() {
        environmentState.print();
        System.out.println("Action: " + action + " Policy Value: " + policyValue + " Reward: " + reward + " State Value: " + stateValue + " TD target: " + tdTarget + " TD error: " + tdError + " Advantage: " + advantage);
    }

    /**
     * Print state chain.
     *
     * @param forward if true prints forward direction otherwise prints backward direction.
     */
    public void print(boolean forward) {
        print();
        if (forward) {
            if (nextState != null) nextState.print(true);
        }
        else {
            if (previousState != null) previousState.print(false);
        }
    }
}
