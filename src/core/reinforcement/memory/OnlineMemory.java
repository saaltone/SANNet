/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.memory;

import utils.DynamicParam;
import utils.DynamicParamException;

import java.io.Serializable;
import java.util.*;

/**
 * Class that defines OnlineMemory.
 *
 */
public class OnlineMemory implements Memory, Serializable {

    private static final long serialVersionUID = 8600974850562595903L;

    /**
     * Random number generator.
     *
     */
    private final Random random = new Random();

    /**
     * Capacity of OnlineMemory.
     *
     */
    private int capacity = 0;

    /**
     * Batch size sampled from OnlineMemory. If batch size is -1 whole memory is sampled.
     *
     */
    private int batchSize = -1;

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
    }

    /**
     * Constructor for OnlineMemory with dynamic parameters.
     *
     * @param params parameters used for OnlineMemory.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public OnlineMemory(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for OnlineMemory.
     *
     * @return parameters used for OnlineMemory.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("capacity", DynamicParam.ParamType.INT);
        paramDefs.put("batchSize", DynamicParam.ParamType.INT);
        return paramDefs;
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
     * Returns size of OnlineMemory.
     *
     */
    public int size() {
        return stateTransitionSet.size();
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
    public TreeSet<StateTransition> getStateTransitions() {
        return sampledStateTransitions;
    }

    /**
     * Returns defined number of random state transitions.
     *
     * @return retrieved state transitions.
     */
    public TreeSet<StateTransition> getRandomStateTransitions() {
        TreeSet<StateTransition> result = new TreeSet<>();
        StateTransition[] sampleArray = (StateTransition[]) stateTransitionSet.toArray();
        for (int sampleIndex = 0; sampleIndex < (batchSize < 0 ? stateTransitionSet.size() : batchSize); sampleIndex++) result.add(sampleArray[random.nextInt(sampleArray.length)]);
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
