/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement;

import core.NeuralNetworkException;
import utils.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashSet;

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
    HashSet<Integer> getAvailableActions();

    /**
     * Takes specific action.
     *
     * @param agent agent that is taking action.
     * @param action action to be taken.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void commitAction(Agent agent, int action) throws NeuralNetworkException, MatrixException, DynamicParamException;

}
