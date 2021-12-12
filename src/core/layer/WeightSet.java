/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer;

import utils.matrix.Matrix;

import java.util.HashSet;

/**
 * Interface that defines weight set for layer.
 *
 */
public interface WeightSet {

    /**
     * Returns set of weights.
     *
     * @return set of weights.
     */
    HashSet<Matrix> getWeights();

    /**
     * Reinitializes weights.
     *
     */
    void reinitialize();

    /**
     * Returns number of parameters.
     *
     * @return number of parameters.
     */
    int getNumberOfParameters();

}
