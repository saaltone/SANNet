/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.ArrayList;

/**
 * Interface for environment.
 *
 */
public interface Environment {

    /**
     * Returns current state of environment.
     *
     * @return state of environment.
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
     */
    int requestAction(Agent agent);

    /**
     * Takes specific action.
     *
     * @param agent agent that is taking action.
     * @param action action to be taken.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void commitAction(Agent agent, int action) throws MatrixException ;

}
