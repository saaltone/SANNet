/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement.policy;

import utils.matrix.Matrix;

import java.util.HashSet;

/**
 * Class that defines GreedyPolicy.
 *
 */
public class GreedyPolicy implements Policy {

    /**
     * Constructor for GreedyPolicy.
     *
     */
    public GreedyPolicy() {
    }

    /**
     * Sets current episode count.
     *
     * @param episodeCount current episode count.
     */
    public void setEpisode(int episodeCount) {}

    /**
     * Takes greedy action.
     *
     * @param stateMatrix current state matrix.
     * @param availableActions available actions in current state
     * @return action taken.
     */
    public int action(Matrix stateMatrix, HashSet<Integer> availableActions) {
        int action = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (Integer row : availableActions) {
            double actionValue = stateMatrix.getValue(row, 0);
            if (maxValue < actionValue || maxValue == Double.NEGATIVE_INFINITY) {
                maxValue =  actionValue;
                action = row;
            }
        }
        return action;
    }

}
