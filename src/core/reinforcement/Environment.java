/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement;

/**
 * Interface defining Environment.
 *
 */
public interface Environment {

    /**
     * Returns true if environment is episodic otherwise false.
     *
     * @return true if environment is episodic otherwise false.
     */
    boolean isEpisodic();

    /**
     * Returns current state of environment.
     *
     * @return state of environment.
     */
    EnvironmentState getState();

    /**
     * Takes specific action in environment.
     *
     * @param agent agent that is taking action.
     * @param action action to be taken.
     */
    void commitAction(Agent agent, int action);

}
