/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.memory;

import utils.DynamicParam;
import utils.DynamicParamException;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeSet;

/**
 * Class that implements prioritized replay memory based on sum tree structure.
 * <br>
 * Reference: https://arxiv.org/pdf/1511.05952.pdf<br>
 *
 */
public class PriorityMemory implements Memory, Serializable {

    private static final long serialVersionUID = -160305763452683523L;

    /**
     * Random number generator.
     *
     */
    private final Random random = new Random();

    /**
     * Capacity of PriorityMemory.
     *
     */
    private int capacity = 20000;

    /**
     * Batch size sampled from PriorityMemory.
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
     * Reference to sum tree.
     *
     */
    private final SumTree sumTree;

    /**
     * If true importance sampling weights are to be applied.
     *
     */
    private boolean applyImportanceSamplingWeights = true;

    /**
     * Sampled state transitions.
     *
     */
    private TreeSet<StateTransition> sampledStateTransitions;

    /**
     * Default constructor for PriorityMemory.
     *
     */
    public PriorityMemory() {
        sumTree = new SumTree(capacity);
    }

    /**
     * Constructor for PriorityMemory with dynamic parameters.
     *
     * @param params parameters used for PriorityMemory.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public PriorityMemory(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
        sumTree = new SumTree(capacity);
    }

    /**
     * Returns parameters used for PriorityMemory.
     *
     * @return parameters used for PriorityMemory.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("capacity", DynamicParam.ParamType.INT);
        paramDefs.put("batchSize", DynamicParam.ParamType.INT);
        paramDefs.put("alpha", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("beta", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("betaStepSize", DynamicParam.ParamType.INT);
        paramDefs.put("applyImportanceSamplingWeights", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for PriorityMemory.<br>
     * <br>
     * Supported parameters are:<br>
     *     - Capacity: capacity of PriorityMemory. Default value 20000.<br>
     *     - batchSize: batch size sampled from PriorityMemory. Default value 32.<br>
     *     - alpha: proportional prioritization factor for samples in PriorityMemory. Default value 0.6.<br>
     *     - beta: term that controls how much prioritization is applied. Default value 0.4.<br>
     *     - betaStepSize: step-size (schedule) which is used to anneal beta value towards 1 (final value). Default value 0.001.<br>
     *     - applyImportanceSamplingWeights: if true importance sampling weights are to be applied during training. Default value true.<br>
     *
     * @param params parameters used for PriorityMemory.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("capacity")) capacity = params.getValueAsInteger("capacity");
        if (params.hasParam("batchSize")) batchSize = params.getValueAsInteger("batchSize");
        if (params.hasParam("alpha")) alpha = params.getValueAsDouble("alpha");
        if (params.hasParam("beta")) beta = params.getValueAsDouble("beta");
        if (params.hasParam("betaStepSize")) betaStepSize = params.getValueAsInteger("betaStepSize");
        if (params.hasParam("applyImportanceSamplingWeights")) applyImportanceSamplingWeights = params.getValueAsBoolean("applyImportanceSamplingWeights");
    }

    /**
     * Returns current size of PriorityMemory.
     *
     */
    public int size() {
        return sumTree.size();
    }

    /**
     * Returns size of sampled set.
     *
     * @return size of sampled set.
     */
    public int sampledSize() {
        return sampledStateTransitions == null ? 0 : sampledStateTransitions.size();
    }

    /**
     * Adds sample into PriorityMemory. Removes old ones exceeding memory capacity by FIFO principle.
     *
     * @param stateTransition state transition to be stored.
     */
    public void add(StateTransition stateTransition) {
        sumTree.add(stateTransition);
    }

    /**
     * Updates state transition into PriorityMemory with new error value.
     *
     * @param stateTransition state transition to be updated.
     */
    public void update(StateTransition stateTransition) {
        double epsilon = 10E-8;
        stateTransition.priority = Math.pow(Math.abs(stateTransition.tdError) + epsilon, alpha);
        sumTree.update(stateTransition);
    }

    /**
     * Updates state transitions in PriorityMemory with new error values.
     *
     * @param stateTransitions state transitions.
     */
    public void update(TreeSet<StateTransition> stateTransitions) {
        for (StateTransition stateTransition : stateTransitions) update(stateTransition);
    }

    /**
     * Samples PriorityMemory.
     *
     */
    public void sample() {
        double entryAmount = sumTree.size();
        double segment = sumTree.getTotalPriority() / (double)batchSize;
        beta = Math.min(beta + betaStepSize, 1);
        double maxWeight = Double.MIN_VALUE;
        sampledStateTransitions = new TreeSet<>();
        for (int sampleIndex = 0; sampleIndex < batchSize; sampleIndex++) {
            double prioritySum = segment * (random.nextDouble() + (double)sampleIndex);
            StateTransition stateTransition = sumTree.getStateTransition(prioritySum);
            if (stateTransition != null) {
                sampledStateTransitions.add(stateTransition);
                maxWeight = Math.max(stateTransition.importanceSamplingWeight = Math.pow(entryAmount * prioritySum, -beta), maxWeight);
            }
        }
        for (StateTransition stateTransition : sampledStateTransitions) stateTransition.importanceSamplingWeight /= maxWeight;
    }

    /**
     * Returns defined number of state transitions from PriorityMemory. Applies priority sampling.
     *
     * @return retrieved state transitions.
     */
    public TreeSet<StateTransition> getStateTransitions() {
        return sampledStateTransitions;
    }

    /**
     * Returns defined number of random state transitions from PriorityMemory.
     *
     * @return retrieved state transitions.
     */
    public TreeSet<StateTransition> getRandomStateTransitions() {
        TreeSet<StateTransition> stateTransitions = new TreeSet<>();
        for (int sampleIndex = 0; sampleIndex < batchSize; sampleIndex++) stateTransitions.add(sumTree.getRandomStateTransition());
        return stateTransitions;
    }

    /**
     * Returns true if memory contains importance sampling weights and they are to be applied otherwise returns false.
     *
     * @return true if memory contains importance sampling weights and they are to be applied otherwise returns false.
     */
    public boolean applyImportanceSamplingWeights() {
        return applyImportanceSamplingWeights;
    }


}
