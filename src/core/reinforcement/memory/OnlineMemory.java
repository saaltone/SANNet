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
     *
     */
    private final static String paramNameTypes = "(capacity:INT)";

    /**
     * Capacity of online memory.
     *
     */
    private int capacity;

    /**
     * Tree set of states in online memory.
     *
     */
    private final TreeSet<State> stateSet = new TreeSet<>();

    /**
     * Sampled states.
     *
     */
    private TreeSet<State> sampledStates;

    /**
     * Default constructor for online memory.
     *
     */
    public OnlineMemory() {
        initializeDefaultParams();
    }

    /**
     * Constructor for online memory.
     *
     * @param capacity capacity.
     */
    private OnlineMemory(int capacity) {
        this.capacity = capacity;
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        capacity = 0;
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
     *
     * @param params parameters used for online memory.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("capacity")) capacity = params.getValueAsInteger("capacity");
    }

    /**
     * Returns reference to memory.
     *
     * @return reference to memory.
     */
    public Memory reference() {
        return new OnlineMemory(capacity);
    }

    /**
     * Adds state into online memory. Removes old ones exceeding memory capacity by FIFO principle.
     *
     * @param state state to be stored.
     */
    public void add(State state) {
        if (stateSet.size() >= capacity && capacity > 0) {
            State removedState = stateSet.pollFirst();
            if (removedState != null) removedState.removePreviousState();
        }
        stateSet.add(state);
    }

    /**
     * Updates states in online memory with new error values.
     *
     * @param states states.
     */
    public void update(TreeSet<State> states) {
    }

    /**
     * Resets memory.
     *
     */
    public void reset() {
        sampledStates = null;
        stateSet.clear();
    }

    /**
     * Samples memory.
     *
     */
    public void sample() {
        if (sampledStates != null) return;
        sampledStates = new TreeSet<>(stateSet);
        stateSet.clear();
    }

    /**
     * Samples defined number of states from online memory.
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
        return false;
    }

}
