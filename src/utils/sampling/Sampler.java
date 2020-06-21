/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.sampling;

import core.NeuralNetworkException;
import utils.Sequence;
import utils.matrix.MatrixException;

/**
 * Interface for Sampler.
 *
 */
public interface Sampler {

    /**
     * Returns depth of sample.
     *
     * @return depth of sample.
     */
    int getDepth();

    /**
     * Resets sampler.
     *
     */
    void reset();

    /**
     * Returns number of training or validation iterations.
     *
     * @return number of training or validation iterations.
     */
    int getNumberOfIterations();

    /**
     * Samples number of samples from input output pairs.
     *
     * @param inputSequence sampled input sequence.
     * @param outputSequence sampled output sequence.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if input and output sequence depths are not equal.
     */
    void getSamples(Sequence inputSequence, Sequence outputSequence) throws MatrixException, NeuralNetworkException;

}
