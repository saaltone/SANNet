/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement.policy;

import utils.matrix.Matrix;

import java.util.HashSet;

/**
 * Interface for Policy.
 *
 */
public interface Policy {

    /**
     * Sets current episode count.
     *
     * @param episodeCount current episode count.
     */
    void setEpisode(int episodeCount);

    /**
     * Takes action.
     *
     * @param stateMatrix current state matrix.
     * @param availableActions available actions in current state
     * @return action taken.
     */
    int action(Matrix stateMatrix, HashSet<Integer> availableActions);

}
