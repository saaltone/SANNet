/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import core.reinforcement.agent.State;
import utils.configurable.Configurable;
import utils.matrix.Matrix;

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
     */
    void increment();

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
     * @param availableActions available actions in current state
     * @param alwaysGreedy if true greedy action is always taken.
     * @return action taken.
     */
    int action(Matrix policyValueMatrix, HashSet<Integer> availableActions, boolean alwaysGreedy);

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
