/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

import core.NeuralNetworkException;
import utils.matrix.MatrixException;

/**
 * Interface for agent.
 *
 */
public interface Agent {

    /**
     * Starts new episode. Resets sequence by default.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void newEpisode() throws NeuralNetworkException, MatrixException;

    /**
     * Starts new episode.
     *
     * @param resetSequence if true resets sequence by putting previous sample to null.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void newEpisode(boolean resetSequence) throws NeuralNetworkException, MatrixException;

    /**
     * Begins new episode step for agent.
     *
     * @param commitPreviousStep if true commits previous step prior starting new step.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void newStep(boolean commitPreviousStep) throws MatrixException;

    /**
     * Commits episode step and adds it into replay buffer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void commitStep() throws MatrixException;

    /**
     * Executes policy taking action with highest value (exploitation) unless random action is selected (exploration).<br>
     * Chooses random policy by probability epsilon or if forced.<br>
     * Requests environment to execute chosen action.<br>
     *
     * @param alwaysGreedy if true greedy action is always taken. ForceRandomAction flag is omitted.
     * @param forceRandomAction if true forces to take valid random action.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void executePolicy(boolean alwaysGreedy, boolean forceRandomAction) throws NeuralNetworkException, MatrixException;

    /**
     * Sets immediate reward for episode step after agent has executed policy.
     *
     * @param reward reward
     */
    void setReward(double reward);

}
