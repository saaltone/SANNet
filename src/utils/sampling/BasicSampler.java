/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.sampling;

import core.NeuralNetworkException;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Sequence;
import utils.matrix.MMatrix;

import java.io.Serializable;
import java.util.*;

/**
 * Class that defines BasicSampler for neural network.<br>
 *
 */
public class BasicSampler implements Sampler, Serializable {

    private static final long serialVersionUID = 1745926046002213714L;

    /**
     * Input sample set for sampling.
     *
     */
    private final LinkedHashMap<Integer, MMatrix> inputs = new LinkedHashMap<>();

    /**
     * Output sample set for sampling.
     *
     */
    private final LinkedHashMap<Integer, MMatrix> outputs = new LinkedHashMap<>();

    /**
     * Number of validation cycles.
     *
     */
    private int numberOfIterations = 1;

    /**
     * If true samples entire input as single set. Default value false.
     *
     */
    private boolean fullSet = false;

    /**
     * If true samples at random start. Default value false.
     *
     */
    private boolean randomStart = false;

    /**
     * If true samples in random order. Default value true.
     *
     */
    private boolean randomOrder = true;

    /**
     * If true sample steps in forward order (not valid for randomOrder sampling). Default value true.
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
     * Depth of sample.
     *
     */
    private int sampleDepth = -1;

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
     * Constructor for BasicSampler.
     *
     * @param inputs input set for sampling.
     * @param outputs output set for sampling.
     * @throws NeuralNetworkException throws exception if input and output set sizes are not equal or not defined.
     */
    public BasicSampler(LinkedHashMap<Integer, MMatrix> inputs, LinkedHashMap<Integer, MMatrix> outputs) throws NeuralNetworkException {
        if (inputs == null || outputs == null) throw new NeuralNetworkException("Inputs or outputs are not defined.");
        if (inputs.isEmpty() || outputs.isEmpty()) throw new NeuralNetworkException("Input and output data sets cannot be empty.");
        if (inputs.size() != outputs.size()) throw new NeuralNetworkException("Size of sample inputs and outputs must match.");
        for (Integer index : inputs.keySet()) addSample(inputs.get(index), outputs.get(index));
        sampleAt = 0;
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
    public BasicSampler(LinkedHashMap<Integer, MMatrix> inputs, LinkedHashMap<Integer, MMatrix> outputs, String params) throws NeuralNetworkException, DynamicParamException {
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
        paramDefs.put("numberOfIterations", DynamicParam.ParamType.INT);
        paramDefs.put("fullSet", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("randomOrder", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("randomStart", DynamicParam.ParamType.BOOLEAN);
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
     *     - numberOfIterations: number of training or validation iterations executed during step. Default value 1.<br>
     *     - fullSet: if true samples entire input as single set. Default value false.<br>
     *     - randomStart: if true samples at random start. Default value false.<br>
     *     - randomOrder: if true samples in random order. Default value true.<br>
     *     - stepForward: if true samples sampling steps in forward order (not valid for randomOrder sampling). Default value true.<br>
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
        if (params.hasParam("numberOfIterations")) {
            numberOfIterations = params.getValueAsInteger("numberOfIterations");
            if (numberOfIterations < 1) throw new DynamicParamException("Number of iterations must be at least 1.");
        }
        if (params.hasParam("fullSet")) fullSet = params.getValueAsBoolean("fullSet");
        if (params.hasParam("randomOrder")) randomOrder = params.getValueAsBoolean("randomOrder");
        if (params.hasParam("randomStart")) randomStart = params.getValueAsBoolean("randomStart");
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
     * Adds sample into sampler.
     *
     * @param input input sample.
     * @param output output sample.
     * @throws NeuralNetworkException throws exception if input or output is not defined.
     */
    private void addSample(MMatrix input, MMatrix output) throws NeuralNetworkException {
        if (input == null) throw new NeuralNetworkException("Input is not defined.");
        if (output == null) throw new NeuralNetworkException("Output is not defined.");
        if (input.size() != output.size()) throw new NeuralNetworkException("Input and output must be same size.");
        if (input.size() == 0) throw new NeuralNetworkException("Input and output cannot be empty.");
        if (input.getCapacity() != output.getCapacity()) throw new NeuralNetworkException("Sample depth of input and output must match.");
        if (sampleDepth == -1) sampleDepth = input.getCapacity();
        else if (sampleDepth != input.getCapacity()) throw new NeuralNetworkException("All input and output samples must have same depth.");
        inputs.put(inputs.size(), input);
        outputs.put(outputs.size(), output);
    }

    /**
     * Returns depth of sample.
     *
     * @return depth of sample.
     */
    public int getDepth() {
        return sampleDepth;
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
     * @return number of training or validation iterations.
     */
    public int getNumberOfIterations() {
        return numberOfIterations;
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
        else maxSampleSize = Math.min(sampleSize, inputs.size());

        if (randomOrder) {
            for (int index = 0; index < maxSampleSize; index++) {
                sampleIndices.add(random.nextInt(inputs.size() - (cyclical ? 1 : maxSampleSize) + 1));
            }
        }
        else {
            if (randomStart) sampleAt = random.nextInt(inputs.size() - (cyclical ? 1 : maxSampleSize) + 1);
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
        }

        if (shuffleSamples && maxSampleSize > 1) Collections.shuffle(sampleIndices);

        for (Integer sampleIndex : sampleIndices) {
            inputSequence.put(sampleIndex, inputs.get(sampleIndex));
            outputSequence.put(sampleIndex, outputs.get(sampleIndex));
        }

        if (!randomOrder && !randomStart && !fullSet) {
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

}
