/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement.policy;

import utils.matrix.Matrix;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Random;

/**
 * Class that defines WeightedRandomPolicy.
 *
 */
public class WeightedRandomPolicy implements Policy, Serializable {

    private static final long serialVersionUID = -5927264161768816361L;

    /**
     * Random function.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for WeightedRandomPolicy.
     *
     */
    public WeightedRandomPolicy() {
    }

    /**
     * Sets current episode count.
     *
     * @param episodeCount current episode count.
     */
    public void setEpisode(int episodeCount) {}

    /**
     * Takes action with WeightedRandomPolicy.
     *
     * @param stateMatrix current state matrix.
     * @param availableActions available actions in current state
     * @return action taken.
     */
    public int action(Matrix stateMatrix, HashSet<Integer> availableActions) {
        int action = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (Integer row : availableActions) {
            double actionValue = stateMatrix.getValue(row, 0) * random.nextGaussian();
            if (maxValue < actionValue || maxValue == Double.NEGATIVE_INFINITY) {
                maxValue =  actionValue;
                action = row;
            }
        }
        return action;
    }

}
