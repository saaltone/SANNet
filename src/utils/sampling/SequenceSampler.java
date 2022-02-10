/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.sampling;

import core.network.NeuralNetworkException;
import utils.configurable.Configurable;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

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
    private final HashMap<Integer, Sequence> inputs = new HashMap<>();

    /**
     * Output sample set for sampling.
     *
     */
    private final HashMap<Integer, Sequence> outputs = new HashMap<>();

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
     * Constructor for sequence sampler.
     *
     * @param inputs input sequences for sampling.
     * @param outputs output sequences for sampling.
     * @throws NeuralNetworkException throws exception if input and output set sizes are not equal or not defined.
     */
    public SequenceSampler(HashMap<Integer, Sequence> inputs, HashMap<Integer, Sequence> outputs) throws NeuralNetworkException {
        initializeDefaultParams();
        if (inputs == null || outputs == null) throw new NeuralNetworkException("Inputs or outputs are not defined.");
        if (inputs.isEmpty() || outputs.isEmpty()) throw new NeuralNetworkException("Input and output data sets cannot be empty.");
        if (inputs.size() != outputs.size()) throw new NeuralNetworkException("Size of sample inputs and outputs must match.");
        for (Map.Entry<Integer, Sequence> entry : inputs.entrySet()) {
            int index = entry.getKey();
            Sequence inputSequence = entry.getValue();
            addSample(inputSequence, outputs.get(index));
        }
        sampleAt = 0;
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
    public SequenceSampler(HashMap<Integer, Sequence> inputs, HashMap<Integer, Sequence> outputs, String params) throws NeuralNetworkException, DynamicParamException {
        this(inputs, outputs);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
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

        Sequence outputSequenceEntry = outputs.get(sampleAt);
        for (Map.Entry<Integer, MMatrix> entry : inputs.get(sampleAt).entrySet()) {
            int sampleIndex = entry.getKey();
            MMatrix input = entry.getValue();
            MMatrix inputSample = new MMatrix(input.getDepth());
            inputSequence.put(sampleIndex, inputSample);
            MMatrix output = outputSequenceEntry.get(sampleIndex);
            MMatrix outputSample = new MMatrix(output.getDepth());
            outputSequence.put(sampleIndex, outputSample);
            for (Map.Entry<Integer, Matrix> depthEntry: input.entrySet()) {
                int depthIndex = depthEntry.getKey();
                Matrix inputMatrix = depthEntry.getValue();
                inputSample.put(depthIndex, inputMatrix);
                outputSample.put(depthIndex, output.get(depthIndex));
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
