/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.memory;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Implements online memory.<br>
 *
 */
public class OnlineMemory implements Memory, Serializable {

    @Serial
    private static final long serialVersionUID = 8600974850562595903L;

    /**
     * Parameter name types for online memory.
     *     - capacity: Capacity of online memory. Default value 0 (unlimited).<br>
     *     - batchSize: Batch size sampled from online memory. Default value -1 (whole memory is sampled).<br>
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
     * Capacity of online memory.
     *
     */
    private int capacity;

    /**
     * Batch size sampled from online memory. If batch size is -1 whole memory is sampled.
     *
     */
    private int batchSize;

    /**
     * Tree set of state transitions in online memory.
     *
     */
    private final TreeSet<StateTransition> stateTransitionSet = new TreeSet<>();

    /**
     * Sampled state transitions.
     *
     */
    private TreeSet<StateTransition> sampledStateTransitions;

    /**
     * Default constructor for online memory.
     *
     */
    public OnlineMemory() {
        initializeDefaultParams();
        params = null;
    }

    /**
     * Default constructor for online memory.
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
     * Returns parameters used for online memory.
     *
     * @return parameters used for online memory.
     */
    public String getParamDefs() {
        return OnlineMemory.paramNameTypes;
    }

    /**
     * Sets parameters used for online memory.<br>
     * <br>
     * Supported parameters are:<br>
     *     - capacity: Capacity of online memory. Default value 0 (unlimited).<br>
     *     - batchSize: Batch size sampled from online memory. Default value -1 (whole memory is sampled).<br>
     *
     * @param params parameters used for online memory.
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
     * Returns size of online memory.
     *
     */
    public int size() {
        return stateTransitionSet.size();
    }

    /**
     * Adds state transition into online memory. Removes old ones exceeding memory capacity by FIFO principle.
     *
     * @param stateTransition state transition to be stored.
     */
    public void add(StateTransition stateTransition) {
        if (stateTransitionSet.size() >= capacity && capacity > 0) stateTransitionSet.pollFirst();
        stateTransitionSet.add(stateTransition);
    }

    /**
     * Updates state transitions in online memory with new error values.
     *
     * @param stateTransitions state transitions.
     */
    public void update(TreeSet<StateTransition> stateTransitions) {
    }

    /**
     * Resets memory.
     *
     */
    public void reset() {
        sampledStateTransitions = null;
        stateTransitionSet.clear();
    }

    /**
     * Samples memory.
     *
     */
    public void sample() {
        sampledStateTransitions = new TreeSet<>(stateTransitionSet);
        stateTransitionSet.clear();
    }

    /**
     * Samples defined number of state transitions from online memory.
     *
     * @return retrieved state transitions.
     */
    public TreeSet<StateTransition> getSampledStateTransitions() {
        return sampledStateTransitions;
    }

    /**
     * Returns true if memory contains importance sampling weights, and they are to be applied otherwise returns false.
     *
     * @return true if memory contains importance sampling weights, and they are to be applied otherwise returns false.
     */
    public boolean applyImportanceSamplingWeights() {
        return false;
    }

    /**
     * Sets if importance sampling weights are applied.
     *
     * @param applyImportanceSamplingWeights if true importance sampling weights are applied otherwise not.
     */
    public void setEnableImportanceSamplingWeights(boolean applyImportanceSamplingWeights) {
    }

}
