/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.sampling;

import java.util.TreeMap;

/**
 * Interface for sampler.<br>
 *
 */
public interface Sampler {

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
     * @param inputSequences sampled input sequences.
     * @param outputSequences sampled output sequences.
     */
    void getSamples(TreeMap<Integer, Sequence> inputSequences, TreeMap<Integer, Sequence>  outputSequences);

}
