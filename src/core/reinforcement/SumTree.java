/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Random;

/**
 * Class that defines SumTree as efficient structure to maintain sample along their priorities and fetch by priority.<br>
 * <br>
 * Reference: https://www.endtoend.ai/deep-rl-seminar/2#prioritized-experience-replay and https://github.com/jaromiru/AI-blog/blob/master/SumTree.py<br>
 *
 */
class SumTree implements Serializable {

    private static final long serialVersionUID = 2643802180771085181L;

    /**
     * Current index of sample.
     *
     */
    private int currentIndex = 0;

    /**
     * Maximum available index of sample defined by capacity of sum tree.
     *
     */
    private final int maxIndex;

    /**
     * Total number of entries in tree.
     *
     */
    private int entries = 0;

    /**
     * Samples stored into sum tree.
     *
     */
    private final RLSample[] samples;

    /**
     * Map that maintain relationship of sample and its index.
     *
     */
    private final HashMap<RLSample, Integer> sampleMap = new HashMap<>();

    /**
     * Array containing structure of sum tree.
     *
     */
    private final double[] sumTree;

    /**
     * Constructor for sum tree.
     *
     * @param capacity capacity of sum tree.
     */
    public SumTree(int capacity) {
        maxIndex = capacity - 1;
        sumTree = new double[2 * capacity - 1];
        samples = new RLSample[capacity];
    }

    /**
     * Adds sample into sum tree into current position and updates sum tree priorities accordingly.
     *
     * @param sample sample to be added into sum tree.
     */
    public void add(RLSample sample) {
        samples[currentIndex] = sample;
        int nodeIndex = currentIndex + maxIndex;
        sampleMap.put(sample, nodeIndex);
        update(sample, nodeIndex);
        currentIndex = currentIndex == maxIndex ? 0 : currentIndex + 1;
        if (entries < maxIndex + 1) entries++;
    }

    /**
     * Updates existing sample in sum tree.
     *
     * @param sample sample to be updated in sum tree.
     */
    public void update(RLSample sample) {
        update (sample, sampleMap.get(sample));
    }

    /**
     * Updates priority of sample in tree.
     *
     * @param sample sample to be updated.
     * @param nodeIndex node index of sample.
     */
    private void update(RLSample sample, int nodeIndex) {
        double priorityDelta = sample.priority - sumTree[nodeIndex];
        sumTree[nodeIndex] = sample.priority;
        propagate (nodeIndex, priorityDelta);
    }

    /**
     * Propagates priority of sample starting from leaf node through parent nodes into root node.
     *
     * @param nodeIndex current node index.
     * @param priorityDelta priority delta of previously stored sample and current sample.
     */
    private void propagate(int nodeIndex, double priorityDelta) {
        int parentIndex = Math.floorDiv(nodeIndex - 1,  2);
        sumTree[parentIndex] += priorityDelta;
        if (parentIndex != 0) propagate(parentIndex, priorityDelta);
    }

    /**
     * Gets sample by given priority sum.
     *
     * @param prioritySum given priority sum.
     * @return fetched samples.
     */
    public RLSample get(double prioritySum) {
        return samples[get(0, prioritySum) - maxIndex];
    }

    /**
     * Gets sample by priority sum. If priority sum is less or equal than current node priority sum search takes left branch otherwise follows right branch.
     *
     * @param nodeIndex current node index.
     * @param prioritySum given priority sum.
     * @return returns node index of search and through recursion finally leading into leaf node.
     */
    private int get(int nodeIndex, double prioritySum) {
        int leftNodeIndex = 2 * nodeIndex + 1;
        int rightNodeIndex = leftNodeIndex + 1;
        if (leftNodeIndex >= sumTree.length) return nodeIndex;
        if (prioritySum <= sumTree[leftNodeIndex] && sumTree[leftNodeIndex] != 0) return get(leftNodeIndex, prioritySum);
        else return get(rightNodeIndex, prioritySum - sumTree[leftNodeIndex]);
    }

    /**
     * Returns total (root) priority sum of sum tree.
     *
     * @return total priority sum of sum tree.
     */
    public double totalPriority() {
        return sumTree[0];
    }

    /**
     * Returns number of entries added to the tree.
     *
     * @return number of entries added to the tree.
     */
    public int getEntries() {
        return entries;
    }

    /**
     * Checks if sum tree already contains sample.
     *
     * @param sample sample to be checked if already in sum tree.
     * @return returns true if sample is already in sum tree otherwise returns false
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public boolean containsSample(RLSample sample) throws MatrixException {
        for (RLSample currentSample : sampleMap.keySet()) if(currentSample.state.equals(sample.state)) return true;
        return false;
    }

    /**
     * Returns random sample.
     *
     * @return random sample.
     */
    public RLSample getRandomSample() {
        return samples[new Random().nextInt(getEntries())];
    }

}
