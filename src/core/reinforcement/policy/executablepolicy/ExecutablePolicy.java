/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.State;
import utils.configurable.Configurable;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashSet;

/**
 * Interface for ExecutablePolicy.<br>
 *
 */
public interface ExecutablePolicy extends Configurable {

    /**
     * Returns parameter definitions for executable policy.
     *
     * @return parameter definitions for executable policy.
     */
    String getParamDefs();

    /**
     * Sets flag if agent is in learning mode.
     *
     * @param isLearning if true agent is in learning mode.
     */
    void setLearning(boolean isLearning);

    /**
     * Increments policy.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void increment() throws MatrixException;

    /**
     * Takes action decided by external agent.
     *
     * @param policyValueMatrix current policy value matrix.
     * @param availableActions available actions in current state
     * @param action action.
     */
    void action(Matrix policyValueMatrix, HashSet<Integer> availableActions, int action);

    /**
     * Takes action based on policy.
     *
     * @param policyValueMatrix current policy value matrix.
     * @param availableActions  available actions in current state
     * @return action taken.
     * @throws AgentException throws exception if policy fails to choose valid action.
     */
    int action(Matrix policyValueMatrix, HashSet<Integer> availableActions) throws AgentException;

    /**
     * Adds state for action execution.
     *
     * @param state state.
     */
    void add(State state);

    /**
     * Ends episode.
     *
     */
    void endEpisode();

    /**
     * Returns executable policy type.
     *
     * @return executable policy type.
     */
    ExecutablePolicyType getExecutablePolicyType();

}
