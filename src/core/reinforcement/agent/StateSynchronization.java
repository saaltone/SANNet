/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.agent;

import java.io.Serial;
import java.io.Serializable;

/**
 * Implements class for state synchronization.
 *
 */
public class StateSynchronization implements Serializable {

    @Serial
    private static final long serialVersionUID = -2505399718616912402L;

    /**
     * Last state.
     *
     */
    private State lastState;

    /**
     * Episode ID.
     *
     */
    private int episodeID;

    /**
     * Time step within episode.
     *
     */
    private int timeStep;

    /**
     * If true new episode is started.
     *
     */
    private boolean newEpisodeStarted = false;

    /**
     * Constructor for state synchronization.
     *
     */
    public StateSynchronization() {
        episodeID = 0;
        timeStep = 0;
        lastState = null;
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
     * Initiates new episode.
     *
     */
    public void newEpisode() {
        if (!newEpisodeStarted) {
            episodeID++;
            timeStep = 0;
            lastState = null;
            newEpisodeStarted = true;
        }
    }

    /**
     * Return next state at new time step.
     *
     * @param environment reference to environment.
     * @return next state.
     */
    public State getNextState(Environment environment) {
        newEpisodeStarted = false;
        timeStep++;
        State state = new State(getEpisodeID(), getTimeStep(), environment.getState());
        if (lastState != null) lastState.nextState = state;
        state.previousState = lastState;
        lastState = state;
        return state;
    }

}
