/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.sampling;

import core.network.NeuralNetworkException;
import utils.configurable.Configurable;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Implements sequence sampler for neural network.<br>
 *
 */
public class SequenceSampler implements Sampler, Configurable, Serializable {

    @Serial
    private static final long serialVersionUID = 4295889925849740870L;

    /**
     * Sets parameters used for sequence sampler.<br>
     * <br>
     * Supported parameters are:<br>
     *     - numberOfIterations: number of training or validation iterations executed during step. Default value 1.<br>
     *     - fullSet: if true sets number of validation of cycles are equal to number of sequences in samples and does not use random sampling. Default value false.<br>
     *     - randomOrder: if true samples in random order. Default value true.<br>
     *     - stepForward: if true samples sampling steps in forward order (not valid for randomOrder sampling). Default value true.<br>
     *     - stepSize: number of steps taken forward or backward when sampling (not valid for randomOrder sampling). Default value 1.<br>
     *
     */
    private final static String paramNameTypes = "(numberOfIterations:iNT), " +
            "(perEpoch:BOOLEAN), " +
            "(fullSet:BOOLEAN), " +
            "(randomOrder:BOOLEAN), " +
            "(stepForward:BOOLEAN), " +
            "(stepSize:INT)";

    /**
     * Input sample set for sampling.
     *
     */
    private final HashMap<Integer, HashMap<Integer, Sequence>> inputs = new HashMap<>();

    /**
     * Output sample set for sampling.
     *
     */
    private final HashMap<Integer, HashMap<Integer, Sequence>> outputs = new HashMap<>();

    /**
     * Number of iterations for training or validation phase.
     *
     */
    private int numberOfIterations;

    /**
     * If true sets number of validation of cycles are equal to number of sequences in samples and does not use random sampling.
     *
     */
    private boolean fullSet;

    /**
     * If true samples in random order. Default value true.
     *
     */
    private boolean randomOrder;

    /**
     * If true sample steps in forward order (no valid for randomOrder sampling). Default value true.
     *
     */
    private boolean stepForward;

    /**
     * Number of steps taken forward or backward when sampling (no valid for randomOrder sampling). Default value 1.
     *
     */
    private int stepSize;

    /**
     * Current sampling position assuming no random sampling.
     *
     */
    private transient int sampleAt;

    /**
     * Random function.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for sequence sampler.
     *
     * @param inputs input sequences for sampling.
     * @param outputs output sequences for sampling.
     * @throws NeuralNetworkException throws exception if input and output set sizes are not equal or not defined.
     */
    public SequenceSampler(HashMap<Integer, HashMap<Integer, Sequence>> inputs, HashMap<Integer, HashMap<Integer, Sequence>> outputs) throws NeuralNetworkException {
        initializeDefaultParams();
        if (inputs == null || outputs == null) throw new NeuralNetworkException("Inputs or outputs are not defined.");
        if (inputs.isEmpty() || outputs.isEmpty()) throw new NeuralNetworkException("Input and output data sets cannot be empty.");
        Set<Integer> sampleIndexSet = null;

        for (Map.Entry<Integer, HashMap<Integer, Sequence>> entry : inputs.entrySet()) {
            HashMap<Integer, Sequence> currentInputMap = new HashMap<>();
            this.inputs.put(entry.getKey(), currentInputMap);
            HashMap<Integer, Sequence> inputsMap = entry.getValue();
            if (sampleIndexSet == null) sampleIndexSet = inputsMap.keySet();
            if (sampleIndexSet.size() != inputsMap.keySet().size()) throw new NeuralNetworkException("Number of samples is not matching");
            currentInputMap.putAll(inputsMap);
        }

        for (Map.Entry<Integer, HashMap<Integer, Sequence>> entry : outputs.entrySet()) {
            HashMap<Integer, Sequence> currentOutputMap = new HashMap<>();
            this.outputs.put(entry.getKey(), currentOutputMap);
            HashMap<Integer, Sequence> outputsMap = entry.getValue();
            if (sampleIndexSet.size() != outputsMap.keySet().size()) throw new NeuralNetworkException("Number of samples is not matching");
            currentOutputMap.putAll(outputsMap);
        }
    }

    /**
     * Constructor for sequence sampler.
     *
     * @param inputs input sequences for sampling.
     * @param outputs output sequences for sampling.
     * @param params parameters used for sequence sampler.
     * @throws NeuralNetworkException throws exception if input and output set sizes are not equal or not defined.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public SequenceSampler(HashMap<Integer, HashMap<Integer, Sequence>> inputs, HashMap<Integer, HashMap<Integer, Sequence>> outputs, String params) throws NeuralNetworkException, DynamicParamException {
        this(inputs, outputs);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        sampleAt = 0;
        numberOfIterations = 1;
        fullSet = false;
        randomOrder = true;
        stepForward = true;
        stepSize = 1;
    }

    /**
     * Returns parameters used for sequence sampler.
     *
     * @return parameters used for sequence sampler.
     */
    public String getParamDefs() {
        return SequenceSampler.paramNameTypes;
    }

    /**
     * Sets parameters used for sequence sampler.<br>
     * <br>
     * Supported parameters are:<br>
     *     - numberOfIterations: number of training or validation iterations executed during step. Default value 1.<br>
     *     - fullSet: if true sets number of validation of cycles are equal to number of sequences in samples and does not use random sampling. Default value false.<br>
     *     - randomOrder: if true samples in random order. Default value true.<br>
     *     - stepForward: if true samples sampling steps in forward order (not valid for randomOrder sampling). Default value true.<br>
     *     - stepSize: number of steps taken forward or backward when sampling (not valid for randomOrder sampling). Default value 1.<br>
     *
     * @param params parameters used for sequence sampler.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("numberOfIterations")) {
            numberOfIterations = params.getValueAsInteger("numberOfIterations");
            if (numberOfIterations < 1) throw new DynamicParamException("Number of iterations must be at least 1.");
        }
        if (params.hasParam("fullSet")) fullSet = params.getValueAsBoolean("fullSet");
        if (params.hasParam("randomOrder")) randomOrder = params.getValueAsBoolean("randomOrder");
        if (params.hasParam("stepForward")) stepForward = params.getValueAsBoolean("stepForward");
        if (params.hasParam("stepSize")) {
            stepSize = params.getValueAsInteger("stepSize");
            if (stepSize < 1) throw new DynamicParamException("Step size must be at least 1.");
        }
    }

    /**
     * Resets sampler.
     *
     */
    public void reset() {
        if (fullSet) sampleAt = 0;
    }

    /**
     * Returns number of training or validation iterations.
     *
     * @return number of training or validation iterations
     */
    public int getNumberOfIterations() {
        return !fullSet ? numberOfIterations : inputs.size();
    }

    /**
     * Samples number of samples from input output pairs.
     *
     * @param inputSequences sampled input sequences.
     * @param outputSequences sampled output sequences.
     */
    public void getSamples(TreeMap<Integer, Sequence> inputSequences, TreeMap<Integer, Sequence>  outputSequences) {
        if (!fullSet && randomOrder) sampleAt = random.nextInt(inputs.size() - 1);

        for (Map.Entry<Integer, HashMap<Integer, Sequence>> entry : inputs.entrySet()) {
            inputSequences.put(entry.getKey(), entry.getValue().get(sampleAt));
        }
        for (Map.Entry<Integer, HashMap<Integer, Sequence>> entry : outputs.entrySet()) {
            outputSequences.put(entry.getKey(), entry.getValue().get(sampleAt));
        }

        if (!randomOrder || fullSet) {
            if (stepForward) {
                sampleAt += stepSize;
                sampleAt = sampleAt > inputs.size() - 1 ? 0 : sampleAt;
            }
            else {
                sampleAt -= stepSize;
                sampleAt = sampleAt < 0  ? inputs.size() - 1 : sampleAt;
            }
        }

    }

}
