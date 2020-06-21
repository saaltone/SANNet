/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement;

import utils.DynamicParam;
import utils.DynamicParamException;

import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeMap;

/**
 * Class that defines OnlineBuffer.
 *
 */
public class OnlineBuffer implements Buffer, Serializable {

    private static final long serialVersionUID = 8600974850562595903L;

    /**
     * Random number generator.
     *
     */
    private final Random random = new Random();

    /**
     * Size of buffer.
     *
     */
    private int size = 0;

    /**
     * Batch size sampled from buffer. If batch size is -1 whole buffer size is sampled.
     *
     */
    private int batchSize = -1;

    /**
     * Queue for samples.
     *
     */
    private ArrayDeque<RLSample> samples = new ArrayDeque<>();

    /**
     * Default constructor for OnlineBuffer.
     *
     */
    public OnlineBuffer() {
    }

    /**
     * Constructor for OnlineBuffer with dynamic parameters.
     *
     * @param params parameters used for OnlineBuffer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public OnlineBuffer(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for OnlineBuffer.
     *
     * @return parameters used for OnlineBuffer.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("size", DynamicParam.ParamType.INT);
        paramDefs.put("batchSize", DynamicParam.ParamType.INT);
        return paramDefs;
    }

    /**
     * Sets parameters used for OnlineBuffer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - size: Size of online buffer. Default value 100.<br>
     *     - batchSize: Batch size sampled from buffer. Default value -1 (whole buffer is sampled).<br>
     *
     * @param params parameters used for OnlineBuffer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("size")) size = params.getValueAsInteger("size");
        if (params.hasParam("batchSize")) batchSize = params.getValueAsInteger("batchSize");
    }

    /**
     * Returns size of buffer.
     *
     */
    public int size() {
        return samples.size();
    }

    /**
     * Adds sample into online buffer. Removes old ones exceeding buffer capacity by FIFO principle.
     *
     * @param sample sample to be stored.
     */
    public void add(RLSample sample) {
        if (samples.size() >= size && size > 0) samples.remove();
        samples.add(sample);
    }

    /**
     * Updates sample in sum tree with new error value.
     *
     * @param sample sample to be updated.
     */
    public void update(RLSample sample) {
    }

    /**
     * Updates samples in buffer with new error values.
     *
     * @param samples samples to be updated.
     */
    public void update(TreeMap<Integer, RLSample> samples) {
    }

    /**
     * Clears buffer.
     *
     */
    public void clear() {
        samples = new ArrayDeque<>();
    }

    /**
     * Retrieves given number of samples from buffer.
     *
     * @return retrieved samples.
     */
    public TreeMap<Integer, RLSample> getSamples() {
        TreeMap<Integer, RLSample> result = new TreeMap<>();
        int index = 0;
        for (RLSample sample : samples) {
            result.put(index++, sample);
            if (index == batchSize && batchSize > 0) break;
        }
        return result;
    }

    /**
     * Returns number of random samples.
     *
     * @return retrieved samples.
     */
    public TreeMap<Integer, RLSample> getRandomSamples() {
        TreeMap<Integer, RLSample> result = new TreeMap<>();
        RLSample[] sampleArray = (RLSample[])samples.toArray();
        for (int sampleIndex = 0; sampleIndex < (batchSize < 0 ? samples.size() : batchSize); sampleIndex++) result.put(sampleIndex, sampleArray[random.nextInt(size)]);
        return result;
    }

    /**
     * Returns true if buffer contains importance sampling weights otherwise returns false.
     *
     * @return true if buffer contains importance sampling weights otherwise returns false.
     */
    public boolean hasImportanceSamplingWeights() {
        return false;
    }

}
