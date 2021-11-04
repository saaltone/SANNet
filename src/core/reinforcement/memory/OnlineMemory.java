/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.memory;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Class that defines OnlineMemory.<br>
 *
 */
public class OnlineMemory implements Memory, Serializable {

    @Serial
    private static final long serialVersionUID = 8600974850562595903L;

    /**
     * Parameter name types for OnlineMemory.
     *     - capacity: Capacity of OnlineMemory. Default value 0 (unlimited).<br>
     *     - batchSize: Batch size sampled from OnlineMemory. Default value -1 (whole memory is sampled).<br>
     *
     */
    private final static String paramNameTypes = "(capacity:INT), " +
            "(batchSize:INT)";

    /**
     * Parameters for memory.
     *
     */
    private final String params;

    /**
     * Random number generator.
     *
     */
    private final Random random = new Random();

    /**
     * Capacity of OnlineMemory.
     *
     */
    private int capacity;

    /**
     * Batch size sampled from OnlineMemory. If batch size is -1 whole memory is sampled.
     *
     */
    private int batchSize;

    /**
     * Tree set of state transitions in OnlineMemory.
     *
     */
    private TreeSet<StateTransition> stateTransitionSet = new TreeSet<>();

    /**
     * Sampled state transitions.
     *
     */
    private TreeSet<StateTransition> sampledStateTransitions;

    /**
     * Default constructor for OnlineMemory.
     *
     */
    public OnlineMemory() {
        initializeDefaultParams();
        params = null;
    }

    /**
     * Default constructor for OnlineMemory.
     *
     * @param params parameters for memory
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public OnlineMemory(String params) throws DynamicParamException {
        initializeDefaultParams();
        this.params = params;
        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        capacity = 0;
        batchSize = -1;
    }

    /**
     * Returns parameters of memory.
     *
     * @return parameters for memory.
     */
    protected String getParams() {
        return params;
    }

    /**
     * Returns parameters used for OnlineMemory.
     *
     * @return parameters used for OnlineMemory.
     */
    public String getParamDefs() {
        return OnlineMemory.paramNameTypes;
    }

    /**
     * Sets parameters used for OnlineMemory.<br>
     * <br>
     * Supported parameters are:<br>
     *     - capacity: Capacity of OnlineMemory. Default value 0 (unlimited).<br>
     *     - batchSize: Batch size sampled from OnlineMemory. Default value -1 (whole memory is sampled).<br>
     *
     * @param params parameters used for OnlineMemory.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("capacity")) capacity = params.getValueAsInteger("capacity");
        if (params.hasParam("batchSize")) batchSize = params.getValueAsInteger("batchSize");
    }

    /**
     * Returns reference to memory.
     *
     * @return reference to memory.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Memory reference() throws DynamicParamException {
        return new OnlineMemory(getParams());
    }

    /**
     * Returns size of OnlineMemory.
     *
     */
    public int size() {
        return stateTransitionSet.size();
    }

    /**
     * Adds state transition into OnlineMemory. Removes old ones exceeding memory capacity by FIFO principle.
     *
     * @param stateTransition state transition to be stored.
     */
    public void add(StateTransition stateTransition) {
        if (stateTransitionSet.size() >= capacity && capacity > 0) stateTransitionSet.pollFirst();
        stateTransitionSet.add(stateTransition);
    }

    /**
     * Updates state transition in OnlineMemory with new error value.
     *
     * @param stateTransition state transition to be updated.
     */
    public void update(StateTransition stateTransition) {
    }

    /**
     * Updates state transitions in OnlineMemory with new error values.
     *
     * @param stateTransitions state transitions.
     */
    public void update(TreeSet<StateTransition> stateTransitions) {
    }

    /**
     * Resets Memory.
     *
     */
    public void reset() {
        stateTransitionSet = new TreeSet<>();
        sampledStateTransitions = null;
    }

    /**
     * Samples memory.
     *
     */
    public void sample() {
        sampledStateTransitions = new TreeSet<>(stateTransitionSet);
        stateTransitionSet = new TreeSet<>();
    }

    /**
     * Samples defined number of state transitions from OnlineMemory.
     *
     * @return retrieved state transitions.
     */
    public TreeSet<StateTransition> getSampledStateTransitions() {
        return sampledStateTransitions;
    }

    /**
     * Returns defined number of random state transitions.
     *
     * @return retrieved state transitions.
     */
    public TreeSet<StateTransition> getRandomStateTransitions() {
        if (stateTransitionSet.isEmpty()) return new TreeSet<>();
        TreeSet<StateTransition> result = new TreeSet<>();
        Object[] sampleArray = stateTransitionSet.toArray();
        int maxIndex = (batchSize < 0 ? stateTransitionSet.size() : batchSize);
        for (int sampleIndex = 0; sampleIndex < maxIndex; sampleIndex++) result.add((StateTransition)sampleArray[random.nextInt(sampleArray.length)]);
        return result;
    }

    /**
     * Returns true if memory contains importance sampling weights and they are to be applied otherwise returns false.
     *
     * @return true if memory contains importance sampling weights and they are to be applied otherwise returns false.
     */
    public boolean applyImportanceSamplingWeights() {
        return false;
    }

}
