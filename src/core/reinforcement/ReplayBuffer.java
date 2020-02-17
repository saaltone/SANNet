/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

import utils.matrix.MatrixException;

import java.util.*;

/**
 * Class that implements replay buffer for reinforcement learning.<br>
 * <br>
 * Reference: https://arxiv.org/pdf/1511.05952.pdf<br>
 *
 */
class ReplayBuffer {

    /**
     * Random number generator.
     *
     */
    private final Random random = new Random();

    /**
     * Size of replay buffer.
     *
     */
    private int size = 2000;

    /**
     * Term which controls shape of sample priority distribution.
     *
     */
    private double alpha = 0.6;

    /**
     * Reference to sum tree maintaining priorities of samples.
     *
     */
    private SumTree sumTree;

    /**
     * Default constructor for replay buffer.
     *
     */
    public ReplayBuffer() {
        sumTree = new SumTree(size);
    }

    /**
     * Constructor for replay buffer with defined size.
     *
     * @param size size of replay buffer.
     */
    public ReplayBuffer(int size) {
        sumTree = new SumTree(size);
    }

    /**
     * Current maximum priority. Used for newly added sample as default priority.
     *
     */
    private double maxPriority = 1;

    /**
     * Sets size of replay buffer.
     *
     * @param size size of replay buffer.
     */
    public void setSize(int size) {
        this.size = size;
        sumTree = new SumTree(size);
    }

    /**
     * Returns size of replay buffer.
     *
     * @return size of replay buffer.
     */
    public int getSize() {
        return size;
    }

    /**
     * Sets alpha for proportional prioritization of replay buffer.
     *
     * @param alpha alpha value for proportional prioritization of replay buffer.
     */
    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    /**
     * Returns alpha for proportional prioritization of replay buffer.
     *
     * @return alpha for proportional prioritization of replay buffer.
     */
    public double getAlpha() {
        return alpha;
    }

    /**
     * Adds sample into replay buffer. Removes old ones exceeding buffer capacity by FIFO principle.
     *
     * @param sample sample to be stored.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void add(Sample sample) throws MatrixException {
        if (sumTree.containsSample(sample)) return;
        sample.priority = maxPriority;
        sumTree.add(sample);
    }

    /**
     * Updates sample in sum tree
     *
     * @param sample sample to be updated.
     */
    public void update(Sample sample) {
        sumTree.update(sample);
    }

    /**
     * Updates priority for sample based on absolute delta with epsilon and alpha adjustment.
     *
     * @param sample sample for which priority is to be calculated and set.
     */
    public void updatePriority(Sample sample, double error) {
        double epsilon = 10E-8;
        sample.priority = Math.pow(Math.abs(error) + epsilon, alpha);
        maxPriority = Math.max(maxPriority, sample.priority);
    }

    /**
     * Retrieves given number of samples from replay buffer. Applies priority sampling.
     *
     * @param sampleAmount amount of samples to be sampled from replay buffer.
     * @return retrieved samples.
     */
    public HashMap<Integer, Sample> getSamples(int sampleAmount) {
        double totalPriority = sumTree.totalPriority();
        double entryAmount = sumTree.getEntries();
        int segment = (int)(totalPriority / (double)sampleAmount);
        HashMap<Integer, Sample> samples = new HashMap<>();
        for (int sampleIndex = 0; sampleIndex < sampleAmount; sampleIndex++) {
            int lowerBound = segment * sampleIndex;
            int upperBound = segment * (sampleIndex + 1);
            double priority = random.nextDouble() * (upperBound - lowerBound + 1) + lowerBound;
            Sample sample = sumTree.get(priority);
            if (sample != null) samples.put(sampleIndex, sample);
        }
        return samples;
    }

    /**
     * Returns number of random samples.
     *
     * @param sampleAmount amount of samples to be sampled from replay buffer.
     * @return retrieved samples.
     */
    public HashMap<Integer, Sample> getRandomSamples(int sampleAmount) {
        HashMap<Integer, Sample> samples = new HashMap<>();
        for (int sampleIndex = 0; sampleIndex < sampleAmount; sampleIndex++) samples.put(sampleIndex, sumTree.getRandomSample());
        return samples;
    }

}
