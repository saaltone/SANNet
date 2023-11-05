/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.memory;

import core.reinforcement.agent.State;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;

import java.io.Serial;
import java.io.Serializable;
import java.util.Random;
import java.util.TreeSet;

/**
 * Implements prioritized replay memory based on search tree structure.<br>
 * <br>
 * Reference: <a href="https://arxiv.org/pdf/1511.05952.pdf">...</a><br>
 *
 */
public class PriorityMemory implements Memory, Serializable {

    @Serial
    private static final long serialVersionUID = -160305763452683523L;

    /**
     * Parameter name types for prioritized replay memory.
     *     - capacity: capacity of prioritized replay memory. Default value 20000.<br>
     *     - batchSize: batch size sampled from prioritized replay memory. Default value 32.<br>
     *     - alpha: proportional prioritization factor for samples in prioritized replay memory. Default value 0.6.<br>
     *     - beta: term that controls how much prioritization is applied. Default value 0.4.<br>
     *     - betaStepSize: step-size (schedule) which is used to anneal beta value towards 1 (final value). Default value 0.001.<br>
     *     - proportionalPrioritization: if true sampling is done proportionally other purely based on priority. Default value true.<br>
     *     - applyImportanceSamplingWeights: if true importance sampling weights are to be applied during training. Default value false.<br>
     *     - applyUniformSampling: if true applies random uniform sampling otherwise applies prioritized sampling. Default value false.<br>
     *
     */
    private final static String paramNameTypes = "(capacity:INT), " +
            "(batchSize:INT), " +
            "(alpha:DOUBLE), " +
            "(beta:DOUBLE), " +
            "(betaStepSize:INT), " +
            "(proportionalPrioritization:BOOLEAN), " +
            "(applyImportanceSamplingWeights:BOOLEAN), " +
            "(applyUniformSampling:BOOLEAN)";

    /**
     * Random number generator.
     *
     */
    private final Random random = new Random();

    /**
     * Capacity of prioritized replay memory.
     *
     */
    private int capacity;

    /**
     * Batch size sampled from prioritized replay memory.
     *
     */
    private int batchSize;

    /**
     * Term which controls shape of sample priority distribution.
     *
     */
    private double alpha;

    /**
     * Term that controls how much prioritization is applied.
     *
     */
    private double beta;

    /**
     * Step-size (schedule) which is used to anneal beta value towards 1 (final value).
     *
     */
    private double betaStepSize;

    /**
     * Reference to search tree.
     *
     */
    private final SearchTree searchTree;

    /**
     * If true sampling is done proportionally based on sample priority otherwise purely based on priority.
     *
     */
    private boolean proportionalPrioritization;

    /**
     * If true importance sampling weights are to be applied.
     *
     */
    private boolean applyImportanceSamplingWeights;

    /**
     * If true importance sampling weights are to be applied.
     *
     */
    private boolean applyUniformSampling;

    /**
     * Sampled states.
     *
     */
    private TreeSet<State> sampledStates;

    /**
     * Default constructor for prioritized replay memory.
     *
     */
    public PriorityMemory() {
        initializeDefaultParams();
        searchTree = new SumTree(capacity);
    }

    /**
     * Constructor for priority memory.
     *
     * @param capacity capacity
     * @param batchSize batch size
     * @param alpha alpha
     * @param beta beta
     * @param betaStepSize beta step size
     * @param proportionalPrioritization if true proportional prioritization if applied otherwise uniform sampling is applied.
     * @param applyImportanceSamplingWeights if true importance sampling weights are applied otherwise not.
     * @param applyUniformSampling if true uniform sampling is applied.
     */
    private PriorityMemory(int capacity, int batchSize, double alpha, double beta, double betaStepSize, boolean proportionalPrioritization, boolean applyImportanceSamplingWeights, boolean applyUniformSampling) {
        this.capacity = capacity;
        this.batchSize = batchSize;
        this.alpha = alpha;
        this.beta = beta;
        this.betaStepSize = betaStepSize;
        this.proportionalPrioritization = proportionalPrioritization;
        this.applyImportanceSamplingWeights = applyImportanceSamplingWeights;
        this.applyUniformSampling = applyUniformSampling;
        searchTree = new SumTree(capacity);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        capacity = 20000;
        batchSize = 32;
        alpha = 0.6;
        beta = 0.4;
        betaStepSize = 0.001;
        proportionalPrioritization = true;
        applyImportanceSamplingWeights = true;
        applyUniformSampling = false;
    }

    /**
     * Returns parameters used for prioritized replay memory.
     *
     * @return parameters used for prioritized replay memory.
     */
    public String getParamDefs() {
        return PriorityMemory.paramNameTypes;
    }

    /**
     * Sets parameters used for prioritized replay memory.<br>
     * <br>
     * Supported parameters are:<br>
     *     - capacity: capacity of prioritized replay memory. Default value 20000.<br>
     *     - batchSize: batch size sampled from prioritized replay memory. Default value 32.<br>
     *     - alpha: proportional prioritization factor for samples in prioritized replay memory. Default value 0.6.<br>
     *     - beta: term that controls how much prioritization is applied. Default value 0.4.<br>
     *     - betaStepSize: step-size (schedule) which is used to anneal beta value towards 1 (final value). Default value 0.001.<br>
     *     - proportionalPrioritization: if true sampling is done proportionally other purely based on priority. Default value true.<br>
     *     - applyImportanceSamplingWeights: if true importance sampling weights are to be applied during training. Default value false.<br>
     *     - applyUniformSampling: if true applies random uniform sampling otherwise applies prioritized sampling. Default value false.<br>
     *
     * @param params parameters used for prioritized replay memory.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("capacity")) capacity = params.getValueAsInteger("capacity");
        if (params.hasParam("batchSize")) batchSize = params.getValueAsInteger("batchSize");
        if (params.hasParam("alpha")) alpha = params.getValueAsDouble("alpha");
        if (params.hasParam("beta")) beta = params.getValueAsDouble("beta");
        if (params.hasParam("betaStepSize")) betaStepSize = params.getValueAsInteger("betaStepSize");
        if (params.hasParam("proportionalPrioritization")) proportionalPrioritization = params.getValueAsBoolean("proportionalPrioritization");
        if (params.hasParam("applyImportanceSamplingWeights")) applyImportanceSamplingWeights = params.getValueAsBoolean("applyImportanceSamplingWeights");
        if (params.hasParam("applyUniformSampling")) applyUniformSampling = params.getValueAsBoolean("applyUniformSampling");
        if (applyUniformSampling) applyImportanceSamplingWeights = false;
    }

    /**
     * Returns reference to memory.
     *
     * @return reference to memory.
     */
    public Memory reference() {
        return new PriorityMemory(capacity, batchSize, alpha, beta, betaStepSize, proportionalPrioritization, applyImportanceSamplingWeights, applyUniformSampling);
    }

    /**
     * Adds sample into prioritized replay memory. Removes old ones exceeding memory capacity by FIFO principle.
     *
     * @param state state to be stored.
     */
    public void add(State state) {
        searchTree.add(state);
    }

    /**
     * Updates state in search tree with new error value.
     *
     * @param state state to be updated.
     */
    private void update(State state) {
        final double epsilon = 10E-8;
        state.priority = Math.pow(Math.abs(state.tdError) + epsilon, alpha);
        searchTree.update(state);
    }

    /**
     * Updates states in prioritized replay memory with new error values.
     *
     * @param states states.
     */
    public void update(TreeSet<State> states) {
        for (State state : states) update(state);
    }

    /**
     * Resets memory.
     *
     */
    public void reset() {
        sampledStates = null;
    }

    /**
     * Samples prioritized replay memory.
     *
     */
    public void sample() {
        sampledStates = new TreeSet<>();
        if (!applyUniformSampling) {
            final double totalPriority = searchTree.getTotalPriority();
            final double segment = totalPriority / (double)batchSize;
            beta = Math.min(beta + betaStepSize, 1);
            double maxWeight = Double.MIN_VALUE;
            for (int sampleIndex = 0; sampleIndex < batchSize; sampleIndex++) {
                double prioritySum = proportionalPrioritization ? segment * random.nextDouble() + (double)sampleIndex : (totalPriority - 10E-8) * random.nextDouble() + 10E-8;
                State state = searchTree.getState(prioritySum);
                if (state != null) {
                    sampledStates.add(state);
                    double weight = state.importanceSamplingWeight = Math.pow(1 / prioritySum, beta);
                    maxWeight = maxWeight == Double.MIN_VALUE ? weight : Math.max(weight, maxWeight);
                }
            }
            for (State state : sampledStates) state.importanceSamplingWeight /= maxWeight;
        }
        else {
            if (searchTree.size() > 0) {
                for (int sampleIndex = 0; sampleIndex < batchSize; sampleIndex++) {
                    sampledStates.add(searchTree.getRandomState());
                }
            }
        }
    }

    /**
     * Returns defined number of states from prioritized replay memory. Applies priority sampling.
     *
     * @return retrieved states.
     */
    public TreeSet<State> getSampledStates() {
        return sampledStates;
    }

    /**
     * Returns true if memory contains importance sampling weights, and they are to be applied otherwise returns false.
     *
     * @return true if memory contains importance sampling weights, and they are to be applied otherwise returns false.
     */
    public boolean applyImportanceSamplingWeights() {
        return applyImportanceSamplingWeights;
    }

}
