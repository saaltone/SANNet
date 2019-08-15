/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

/**
 * Class that implements efficient structure to maintain sample along their priorities and fetch by priority.<br>
 * <br>
 * Reference: https://www.endtoend.ai/deep-rl-seminar/2#prioritized-experience-replay and https://github.com/jaromiru/AI-blog/blob/master/SumTree.py<br>
 *
 */
public class SumTree {

    /**
     * Current index of sample.
     *
     */
    private int currentIndex = 0;

    /**
     * Maximum available index of sample defined by capacity of sum tree.
     *
     */
    private int maxIndex;

    /**
     * Samples stored into sum tree.
     *
     */
    private Sample[] samples;

    /**
     * Array containing structure of sum tree.
     *
     */
    private double[] sumTree;

    /**
     * Constructor for sum tree.
     *
     * @param capacity capacity of sum tree.
     */
    public SumTree(int capacity) {
        maxIndex = capacity - 1;
        sumTree = new double[2 * capacity - 1];
        samples = new Sample[capacity];
    }

    /**
     * Adds sample into sum tree into current position and updates sum tree priorities accordingly.
     *
     * @param sample sample to be added into sum tree.
     */
    public void add(Sample sample) {
        samples[currentIndex] = sample;
        int nodeIndex = currentIndex + maxIndex;
        double priorityDelta = sample.priority - sumTree[nodeIndex];
        sumTree[nodeIndex] = sample.priority;
        propagate (nodeIndex, priorityDelta);
        currentIndex = currentIndex == maxIndex ? 0 : currentIndex + 1;
    }

    /**
     * Propagates priority of sample starting from leaf node through parent nodes into root node.
     *
     * @param nodeIndex current node index.
     * @param priorityDelta priority delta of previously stored sample and current sample.
     */
    private void propagate(int nodeIndex, double priorityDelta) {
        int parentIndex = (nodeIndex - 1) / 2;
        sumTree[parentIndex] += priorityDelta;
        if (parentIndex != 0) propagate(parentIndex, priorityDelta);
    }

    /**
     * Gets sample by given priority sum.
     *
     * @param prioritySum given priority sum.
     * @return fetched samples.
     */
    public Sample get(double prioritySum) {
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
        int rightNodeIndex = 2 * nodeIndex + 2;
        if (leftNodeIndex >= sumTree.length) return nodeIndex;
        if (prioritySum <= sumTree[leftNodeIndex]) return get(leftNodeIndex, prioritySum);
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

}
