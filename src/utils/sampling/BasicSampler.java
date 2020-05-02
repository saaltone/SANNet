/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.sampling;

import core.NeuralNetworkException;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Sample;
import utils.Sequence;

import java.io.Serializable;
import java.util.*;

/**
 * Class that defines BasicSampler for neural network.
 *
 */
public class BasicSampler implements Sampler, Serializable {

    private static final long serialVersionUID = 1745926046002213714L;

    /**
     * Input sample set for sampling.
     *
     */
    private transient LinkedHashMap<Integer, Sample> inputs;

    /**
     * Output sample set for sampling.
     *
     */
    private transient LinkedHashMap<Integer, Sample> outputs;

    /**
     * Number of validation cycles.
     *
     */
    private int numberOfValidationCycles = 1;

    /**
     * If true samples entire input as single set. Default value false.
     *
     */
    private boolean fullSet = false;

    /**
     * If true samples in random order. Default value true.
     *
     */
    private boolean randomOrder = true;

    /**
     * If true sample steps in forward order (no valid for randomOrder sampling). Default value true.
     *
     */
    private boolean stepForward = true;

    /**
     * Number of steps taken forward or backward when sampling (no valid for randomOrder sampling). Default value 1.
     *
     */
    private int stepSize = 1;

    /**
     * If true shuffles sampled samples. Default value false.
     *
     */
    private boolean shuffleSamples = false;

    /**
     * If true samples in reverse order (assumes no sample shuffling). Default value false.
     *
     */
    private boolean sampleReverse = false;

    /**
     * Number of samples sampled. Default value 1.
     *
     */
    private int sampleSize = 1;

    /**
     * If true considers input samples as cyclical. Default value false.
     *
     */
    private boolean cyclical = false;

    /**
     * Current sampling position assuming no random sampling.
     *
     */
    private int sampleAt = 0;

    /**
     * Random function.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for BasicSampler.
     *
     * @param inputs input set for sampling.
     * @param outputs output set for sampling.
     * @throws NeuralNetworkException throws exception if input and output set sizes are not equal or not defined.
     */
    public BasicSampler(LinkedHashMap<Integer, Sample> inputs, LinkedHashMap<Integer, Sample> outputs) throws NeuralNetworkException {
        initialize(inputs, outputs);
    }

    /**
     * Constructor for BasicSampler.
     *
     * @param params parameters used for BasicSampler.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public BasicSampler(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Constructor for BasicSampler.
     *
     * @param inputs input set for sampling.
     * @param outputs output set for sampling.
     * @param params parameters used for BasicSampler.
     * @throws NeuralNetworkException throws exception if input and output set sizes are not equal or not defined.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public BasicSampler(LinkedHashMap<Integer, Sample> inputs, LinkedHashMap<Integer, Sample> outputs, String params) throws NeuralNetworkException, DynamicParamException {
        this(inputs, outputs);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for BasicSampler.
     *
     * @return parameters used for BasicSampler.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("numberOfValidationCycles", DynamicParam.ParamType.INT);
        paramDefs.put("fullSet", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("randomOrder", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("stepForward", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("stepSize", DynamicParam.ParamType.INT);
        paramDefs.put("shuffleSamples", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("sampleReverse", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("sampleSize", DynamicParam.ParamType.INT);
        paramDefs.put("cyclical", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for BasicSampler.<br>
     * <br>
     * Supported parameters are:<br>
     *     - numberOfValidationCycles: number of validation cycles executed during validation step. Default value 1.<br>
     *     - fullSet: if true samples entire input as single set. Default value false.<br>
     *     - randomOrder: if true samples in random order. Default value true.<br>
     *     - stepForward: if true samples sampling stept in forward order (not valid for randomOrder sampling). Default value true.<br>
     *     - stepSize: number of steps taken forward or backward when sampling (not valid for randomOrder sampling). Default value 1.<br>
     *     - shuffleSamples: if true shuffles sampled samples. Default value false.<br>
     *     - sampleReverse: if true samples in reverse order (assumes no sample shuffling). Default value false.<br>
     *     - sampleSize: number of samples sampled. Default value 1.<br>
     *     - cyclical: if true considered sample set as cyclical. Default value false.<br>
     *
     * @param params parameters used for BasicSampler.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("numberOfValidationCycles")) {
            numberOfValidationCycles = params.getValueAsInteger("numberOfValidationCycles");
            if (numberOfValidationCycles < 1) throw new DynamicParamException("Number of validation cycles must be at least 1.");
        }
        if (params.hasParam("fullSet")) fullSet = params.getValueAsBoolean("fullSet");
        if (params.hasParam("randomOrder")) randomOrder = params.getValueAsBoolean("randomOrder");
        if (params.hasParam("stepForward")) stepForward = params.getValueAsBoolean("stepForward");
        if (params.hasParam("stepSize")) {
            stepSize = params.getValueAsInteger("stepSize");
            if (stepSize < 1) throw new DynamicParamException("Step size must be at least 1.");
        }
        if (params.hasParam("shuffleSamples")) shuffleSamples = params.getValueAsBoolean("shuffleSamples");
        if (params.hasParam("sampleReverse")) sampleReverse = params.getValueAsBoolean("sampleReverse");
        if (params.hasParam("sampleSize")) {
            sampleSize = params.getValueAsInteger("sampleSize");
            if (sampleSize < 1) throw new DynamicParamException("Sample size must be at least 1.");
        }
        if (params.hasParam("cyclical")) cyclical = params.getValueAsBoolean("cyclical");
    }

    /**
     * Initializes sampler.
     *
     * @param inputs input set for sampling.
     * @param outputs output set for sampling.
     * @throws NeuralNetworkException throws exception if input and output set sizes are not equal or not defined.
     */
    private void initialize(LinkedHashMap<Integer, Sample> inputs, LinkedHashMap<Integer, Sample> outputs) throws NeuralNetworkException {
        if (inputs == null || outputs == null) throw new NeuralNetworkException("Inputs or outputs are not defined.");
        if (inputs.isEmpty() || outputs.isEmpty()) throw new NeuralNetworkException("Input and output data sets cannot be empty.");
        if (inputs.size() != outputs.size()) throw new NeuralNetworkException("Size of sample inputs and outputs must match.");
        this.inputs = new LinkedHashMap<>();
        this.outputs = new LinkedHashMap<>();
        for (Integer index : inputs.keySet()) addSample(inputs.get(index), outputs.get(index));
        sampleAt = 0;
    }

    /**
     * Adds sample into sampler.
     *
     * @param input input sample.
     * @param output output sample.
     * @throws NeuralNetworkException throws exception if input or output is not defined.
     */
    private void addSample(Sample input, Sample output) throws NeuralNetworkException {
        if (input == null) throw new NeuralNetworkException("Input is not defined.");
        if (output == null) throw new NeuralNetworkException("Input is not defined.");
        inputs.put(inputs.size(), input);
        outputs.put(outputs.size(), output);
    }

    /**
     * Returns number of validation cycles.
     *
     * @return number of validation cycles.
     */
    public int getNumberOfValidationCycles() {
        return numberOfValidationCycles;
    }

    /**
     * Samples number of samples from input output pairs.
     *
     * @param inputSequence sampled input sequence.
     * @param outputSequence sampled output sequence.
     * @throws NeuralNetworkException throws exception if input and output sequence depths are not equal.
     */
    public void getSamples(Sequence inputSequence, Sequence outputSequence) throws NeuralNetworkException {
        if (inputSequence.getDepth() != outputSequence.getDepth()) throw new NeuralNetworkException("Depth of samples input and output sequences must match");

        ArrayList<Integer> sampleIndices = new ArrayList<>();

        int maxSampleSize;
        if (fullSet) {
            sampleAt = 0;
            maxSampleSize = inputs.size();
        }
        else {
            maxSampleSize = Math.min(sampleSize, inputs.size());
            if (randomOrder) sampleAt = random.nextInt(inputs.size() - (cyclical ? 1 : maxSampleSize) + 1);
        }

        int sampleAtIndex = !fullSet ? sampleAt : 0;
        for (int index = 0; index < maxSampleSize; index++) {
            sampleIndices.add(sampleAtIndex);
            if (sampleReverse) {
                sampleAtIndex--;
                sampleAtIndex = sampleAtIndex < 0 ? inputs.size() - 1 : sampleAtIndex;
            }
            else {
                sampleAtIndex++;
                sampleAtIndex = sampleAtIndex > inputs.size() - 1 ? 0 : sampleAtIndex;
            }
        }

        if (shuffleSamples && maxSampleSize > 1) Collections.shuffle(sampleIndices);

        for (Integer sampleIndex : sampleIndices) {
            inputSequence.put(sampleIndex, inputs.get(sampleIndex));
            outputSequence.put(sampleIndex, outputs.get(sampleIndex));
        }

        if (!randomOrder && !fullSet) {
            if (stepForward) {
                sampleAt += stepSize;
                sampleAt = sampleAt > inputs.size() - (cyclical ? 1 : maxSampleSize) ? 0 : sampleAt;
            }
            else {
                sampleAt -= stepSize;
                sampleAt = sampleAt < 0  ? inputs.size() - (cyclical ? 1 : maxSampleSize) : sampleAt;
            }
        }

    }

    /**
     * Resets sampler.
     *
     */
    public void reset() {
        sampleAt = 0;
    }

}
