/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

import utils.DynamicParam;
import utils.DynamicParamException;

import java.io.Serializable;
import java.util.*;

/**
 * Class that implements replay buffer for reinforcement learning.<br>
 * <br>
 * Reference: https://arxiv.org/pdf/1511.05952.pdf<br>
 *
 */
public class ReplayBuffer implements Buffer, Serializable {

    private static final long serialVersionUID = 4946615850779048129L;

    /**
     * Random number generator.
     *
     */
    private final Random random = new Random();

    /**
     * Size of replay buffer.
     *
     */
    private int size = 20000;

    /**
     * Batch size sampled from buffer.
     *
     */
    private int batchSize = 32;

    /**
     * Term which controls shape of sample priority distribution.
     *
     */
    private double alpha = 0.6;

    /**
     * Term that controls how much prioritization is applied.
     *
     */
    private double beta = 0.4;

    /**
     * Step-size (schedule) which is used to anneal beta value towards 1 (final value).
     *
     */
    private double betaStepSize = 0.001;

    /**
     * Reference to sum tree maintaining priorities of samples.
     *
     */
    private final SumTree sumTree;

    /**
     * Current maximum priority. Used for newly added sample as default priority.
     *
     */
    private double maxPriority = 1;

    /**
     * Default constructor for replay buffer.
     *
     */
    public ReplayBuffer() {
        sumTree = new SumTree(size);
    }

    /**
     * Constructor for ReplayBuffer with dynamic parameters.
     *
     * @param params parameters used for ReplayBuffer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ReplayBuffer(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
        sumTree = new SumTree(size);
    }

    /**
     * Returns parameters used for ReplayBuffer.
     *
     * @return parameters used for ReplayBuffer.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("size", DynamicParam.ParamType.INT);
        paramDefs.put("batchSize", DynamicParam.ParamType.INT);
        paramDefs.put("alpha", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("beta", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("betaStepSize", DynamicParam.ParamType.INT);
        return paramDefs;
    }

    /**
     * Sets parameters used for ReplayBuffer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - size: Size of replay buffer. Default value 20000.<br>
     *     - batchSize: Batch size sampled from buffer. Default value 100.<br>
     *     - alpha: proportional prioritization factor for samples in replay buffer. Default value 0.6.<br>
     *     - beta: term that controls how much prioritization is applied. Default value 0.4.<br>
     *     - betaStepSize: step-size (schedule) which is used to anneal beta value towards 1 (final value). Default value 0.001.<br>
     *
     * @param params parameters used for ReplayBuffer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("size")) size = params.getValueAsInteger("size");
        if (params.hasParam("batchSize")) batchSize = params.getValueAsInteger("batchSize");
        if (params.hasParam("alpha")) alpha = params.getValueAsDouble("alpha");
        if (params.hasParam("beta")) beta = params.getValueAsDouble("beta");
        if (params.hasParam("betaStepSize")) betaStepSize = params.getValueAsInteger("betaStepSize");
    }

    /**
     * Returns size of buffer.
     *
     */
    public int size() {
        return sumTree.getEntries();
    }

    /**
     * Adds sample into replay buffer. Removes old ones exceeding buffer capacity by FIFO principle.
     *
     * @param sample sample to be stored.
     */
    public void add(RLSample sample) {
        sample.priority = maxPriority;
        sumTree.add(sample);
    }

    /**
     * Updates sample in sum tree with new error value.
     *
     * @param sample sample to be updated.
     */
    public void update(RLSample sample) {
        double epsilon = 10E-8;
        sample.priority = Math.pow(Math.abs(sample.tdError) + epsilon, alpha);
        maxPriority = Math.max(maxPriority, sample.priority);
        sumTree.update(sample);
    }

    /**
     * Updates samples in buffer with new error values.
     *
     * @param samples samples to be updated.
     */
    public void update(TreeMap<Integer, RLSample> samples) {
        for (RLSample sample : samples.values()) update(sample);
    }

    /**
     * Clears buffer.
     *
     */
    public void clear() {
    }

    /**
     * Retrieves given number of samples from ReplayBuffer. Applies priority sampling.
     *
     * @return retrieved samples.
     */
    public TreeMap<Integer, RLSample> getSamples() {
        double totalPriority = sumTree.totalPriority();
        double entryAmount = sumTree.getEntries();
        double segment = totalPriority / (double)batchSize;
        beta = Math.min(beta + betaStepSize, 1);
        double maxWeight = Double.NEGATIVE_INFINITY;
        TreeMap<Integer, RLSample> samples = new TreeMap<>();
        int currentIndex = 0;
        for (int sampleIndex = 0; sampleIndex < batchSize; sampleIndex++) {
            double lowerBound = segment * sampleIndex;
            double upperBound = segment * (sampleIndex + 1);
            double priority = random.nextDouble() * (upperBound - lowerBound + 1) + lowerBound;
            RLSample sample = sumTree.get(priority);
            if (sample != null) {
                samples.put(currentIndex++, sample);
                sample.importanceSamplingWeight = Math.pow(entryAmount * priority, -beta);
                maxWeight = Math.max(sample.importanceSamplingWeight, maxWeight);
            }
        }
        for (RLSample sample : samples.values()) sample.importanceSamplingWeight /= maxWeight;
        return samples;
    }

    /**
     * Returns number of random samples.
     *
     * @return retrieved samples.
     */
    public TreeMap<Integer, RLSample> getRandomSamples() {
        TreeMap<Integer, RLSample> samples = new TreeMap<>();
        for (int sampleIndex = 0; sampleIndex < batchSize; sampleIndex++) samples.put(sampleIndex, sumTree.getRandomSample());
        return samples;
    }

    /**
     * Returns true if buffer contains importance sampling weights otherwise returns false.
     *
     * @return true if buffer contains importance sampling weights otherwise returns false.
     */
    public boolean hasImportanceSamplingWeights() {
        return true;
    }

}
