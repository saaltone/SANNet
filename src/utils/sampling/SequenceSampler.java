/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.sampling;

import core.NeuralNetworkException;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Sequence;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.*;

/**
 * Class that defines SequenceSampler for neural network.<br>
 *
 */
public class SequenceSampler implements Sampler, Serializable {

    private static final long serialVersionUID = 4295889925849740870L;

    /**
     * Input sample set for sampling.
     *
     */
    private final LinkedHashMap<Integer, Sequence> inputs = new LinkedHashMap<>();

    /**
     * Output sample set for sampling.
     *
     */
    private final LinkedHashMap<Integer, Sequence> outputs = new LinkedHashMap<>();

    /**
     * Number of iterations for training or validation phase.
     *
     */
    private int numberOfIterations = 1;

    /**
     * If true sets number of validation of cycles are equal to number of sequences in samples and does not use random sampling.
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
     * Depth of sample.
     *
     */
    private int sampleDepth = -1;

    /**
     * Current sampling position assuming no random sampling.
     *
     */
    private transient int sampleAt = 0;

    /**
     * Random function.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for SequenceSampler.
     *
     * @param inputs input sequences for sampling.
     * @param outputs output sequences for sampling.
     * @throws NeuralNetworkException throws exception if input and output set sizes are not equal or not defined.
     */
    public SequenceSampler(LinkedHashMap<Integer, Sequence> inputs, LinkedHashMap<Integer, Sequence> outputs) throws NeuralNetworkException {
        if (inputs == null || outputs == null) throw new NeuralNetworkException("Inputs or outputs are not defined.");
        if (inputs.isEmpty() || outputs.isEmpty()) throw new NeuralNetworkException("Input and output data sets cannot be empty.");
        if (inputs.size() != outputs.size()) throw new NeuralNetworkException("Size of sample inputs and outputs must match.");
        for (Integer index : inputs.keySet()) addSample(inputs.get(index), outputs.get(index));
    }

    /**
     * Constructor for SequenceSampler.
     *
     * @param inputs input sequences for sampling.
     * @param outputs output sequences for sampling.
     * @param params parameters used for SequenceSampler.
     * @throws NeuralNetworkException throws exception if input and output set sizes are not equal or not defined.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public SequenceSampler(LinkedHashMap<Integer, Sequence> inputs, LinkedHashMap<Integer, Sequence> outputs, String params) throws NeuralNetworkException, DynamicParamException {
        this(inputs, outputs);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for SequenceSampler.
     *
     * @return parameters used for SequenceSampler.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("numberOfIterations", DynamicParam.ParamType.INT);
        paramDefs.put("fullSet", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("randomOrder", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("stepForward", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("stepSize", DynamicParam.ParamType.INT);
        return paramDefs;
    }

    /**
     * Sets parameters used for SequenceSampler.<br>
     * <br>
     * Supported parameters are:<br>
     *     - numberOfIterations: number of training or validation iterations executed during step. Default value 1.<br>
     *     - fullSet: if true sets number of validation of cycles are equal to number of sequences in samples and does not use random sampling. Default value false.<br>
     *     - randomOrder: if true samples in random order. Default value true.<br>
     *     - stepForward: if true samples sampling steps in forward order (not valid for randomOrder sampling). Default value true.<br>
     *     - stepSize: number of steps taken forward or backward when sampling (not valid for randomOrder sampling). Default value 1.<br>
     *
     * @param params parameters used for SequenceSampler.
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
     * Adds sample into sampler.
     *
     * @param input input sequence.
     * @param output output sequence.
     * @throws NeuralNetworkException throws exception if input or output is not defined.
     */
    private void addSample(Sequence input, Sequence output) throws NeuralNetworkException {
        if (input == null) throw new NeuralNetworkException("Input is not defined.");
        if (output == null) throw new NeuralNetworkException("Output is not defined.");
        if (input.totalSize() != output.totalSize()) throw new NeuralNetworkException("Input and output must be same size.");
        if (input.totalSize() == 0) throw new NeuralNetworkException("Input and output cannot be empty.");
        if (input.getDepth() != output.getDepth()) throw new NeuralNetworkException("Sample depth of input and output must match.");
        if (sampleDepth == -1) sampleDepth = input.getDepth();
        else if (sampleDepth != input.getDepth()) throw new NeuralNetworkException("All input and output samples must have same depth.");
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
     * @return number of training or validation iterations
     */
    public int getNumberOfIterations() {
        return !fullSet ? numberOfIterations : inputs.size();
    }

    /**
     * Samples number of samples from input output pairs.
     *
     * @param inputSequence sampled input sequence.
     * @param outputSequence sampled output sequence.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if input and output sequence depths are not equal.
     */
    public void getSamples(Sequence inputSequence, Sequence outputSequence) throws MatrixException, NeuralNetworkException {
        if (inputSequence.getDepth() != outputSequence.getDepth()) throw new NeuralNetworkException("Depth of samples input and output sequences must match");

        if (!fullSet && randomOrder) sampleAt = random.nextInt(inputs.size() - 1);

        for (Integer sampleIndex : inputs.get(sampleAt).keySet()) {
            for (Integer entryIndex : inputs.get(sampleAt).get(sampleIndex).keySet()) {
                inputSequence.put(sampleIndex, entryIndex, inputs.get(sampleAt).get(sampleIndex).get(entryIndex));
                outputSequence.put(sampleIndex, entryIndex, outputs.get(sampleAt).get(sampleIndex).get(entryIndex));
            }
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
