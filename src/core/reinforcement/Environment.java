/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

import utils.Matrix;

import java.util.ArrayList;

/**
 * Class that implements interface for environment.
 *
 */
public interface Environment {

    /**
     * Returns current state of environment for the agent.
     *
     * @return state of environment
     */
    Matrix getState();

    /**
     * True if state is terminal. This is usually true if episode is completed.
     *
     * @return true if state is terminal.
     */
    boolean isTerminalState();

    /**
     * Returns available actions in current state of environment
     *
     * @return available actions in current state of environment.
     */
    ArrayList<Integer> getAvailableActions();

    /**
     * Checks if action is valid.
     *
     * @param agent agent that is taking action.
     * @param action action to be taken.
     * @return true if action can be taken successfully.
     */
    boolean isValidAction(Agent agent, int action);

    /**
     * Requests (random) action defined by environment.
     *
     * @param agent agent that is taking action.
     * @return action taken
     * @throws AgentException throws exception if action was not in list of available ones.
     */
    int requestAction(Agent agent) throws AgentException;

    /**
     * Takes specific action.
     *
     * @param agent agent that is taking action.
     * @param action action to be taken.
     * @throws AgentException throws exception if action was not in list of available ones.
     */
    void commitAction(Agent agent, int action) throws AgentException;

    /**
     * Requests immediate reward from environment after taking action.
     *
     * @param agent agent that is asking for reward.
     * @param validAction true if taken action was available one.
     * @return immediate reward.
     */
    double requestReward(Agent agent, boolean validAction);

}
