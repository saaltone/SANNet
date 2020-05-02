/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

import java.util.TreeMap;

/**
 * Interface for Buffer.
 *
 */
public interface Buffer {

    /**
     * Returns size of buffer.
     *
     * @return size of buffer.
     */
    int size();

    /**
     * Adds sample into buffer.
     *
     * @param sample sample to be stored.
     */
    void add(RLSample sample);

    /**
     * Updates sample in buffer with new error value.
     *
     * @param sample sample to be updated.
     */
    void update(RLSample sample);

    /**
     * Updates samples in buffer with new error values.
     *
     * @param samples samples to be updated.
     */
    void update(TreeMap<Integer, RLSample> samples);

    /**
     * Clears buffer.
     *
     */
    void clear();

    /**
     * Retrieves given number of samples from buffer.
     *
     * @return retrieved samples.
     */
    TreeMap<Integer, RLSample> getSamples();

    /**
     * Returns number of random samples.
     *
     * @return retrieved samples.
     */
    TreeMap<Integer, RLSample> getRandomSamples();

    /**
     * Returns true if buffer contains importance sampling weights otherwise returns false.
     *
     * @return true if buffer contains importance sampling weights otherwise returns false.
     */
    boolean hasImportanceSamplingWeights();

}
