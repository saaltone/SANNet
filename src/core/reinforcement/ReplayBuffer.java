package core.reinforcement;

import utils.Matrix;
import utils.MatrixException;

import java.util.*;

/**
 * Class that implements replay buffer for reinforcement learning.<br>
 * <br>
 * Reference: https://arxiv.org/pdf/1511.05952.pdf<br>
 *
 */
public class ReplayBuffer {

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
     * Epsilon term avoids zero priority values.<br>
     *
     */
    private final double epsilon = 10E-8;

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
     * Sets size of replay buffer.
     *
     * @param size size of replay buffer.
     */
    public void setSize(int size) {
        this.size = size;
        sumTree = new SumTree(size);
    }

    /**
     * Gets size of replay buffer.
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
     * Gets alpha for proportional prioritization of replay buffer.
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
    public void add(Sample sample) {
        setPriority(sample);
        sumTree.add(sample);
    }

    /**
     * Sets priority for sample based on absolute delta with epsilon and alpha adjustment.
     *
     * @param sample sample for which priority is to be calculated and set.
     */
    private void setPriority(Sample sample) {
        sample.priority = Math.pow(Math.abs(sample.delta) + epsilon, alpha);
    }

    /**
     * Retrieves given number of samples from replay buffer. Applies priority sampling.
     *
     * @param sampleAmount amount of samples to be sampled from replay buffer.
     * @param states structure that stores states of samples.
     * @param values structure that stores values of samples.
     */
    public void getSamples(int sampleAmount, LinkedHashMap<Integer, Matrix> states, LinkedHashMap<Integer, Matrix> values) {
        double segment = sumTree.totalPriority() / (double)sampleAmount;
        for (int sampleIndex = 0; sampleIndex < sampleAmount; sampleIndex++) {
            double proba = segment * (random.nextDouble() + sampleIndex);
            Sample sample = sumTree.get(proba);
            if (sample != null) {
                states.put(sampleIndex, sample.state);
                values.put(sampleIndex, sample.values);
            }
        }
    }

}
