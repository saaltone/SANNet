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
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Implements basic sampler for neural network.<br>
 *
 */
public class BasicSampler implements Sampler, Configurable, Serializable {

    @Serial
    private static final long serialVersionUID = 1745926046002213714L;

    /**
     * Sets parameters used for basic sampler.<br>
     * <br>
     * Supported parameters are:<br>
     *     - numberOfIterations: number of training or validation iterations executed during step. Default value 1.<br>
     *     - perEpoch: if true sampling takes place epoch wise i.e. samples are removed from sample set until all samples have been sampled at least once. Default value false.<br>
     *     - fullSet: if true samples entire input as single set. Default value false.<br>
     *     - randomStart: if true samples at random start. Default value false.<br>
     *     - randomStartAfterSteps: if greater than zero chooses random start after defined steps. Default value -1.<br>
     *     - randomOrder: if true samples in random order. Default value true.<br>
     *     - stepForward: if true samples sampling steps in forward order (not valid for randomOrder sampling). Default value true.<br>
     *     - stepSize: number of steps taken forward or backward when sampling (not valid for randomOrder sampling). Default value 1.<br>
     *     - shuffleSamples: if true shuffles sampled samples. Default value false.<br>
     *     - sampleReverse: if true samples in reverse order (assumes no sample shuffling). Default value false.<br>
     *     - sampleSize: number of samples sampled. Default value 1.<br>
     *     - cyclical: if true considered sample set as cyclical. Default value false.<br>
     *
     */
    private final static String paramNameTypes = "(numberOfIterations:iNT), " +
            "(perEpoch:BOOLEAN), " +
            "(fullSet:BOOLEAN), " +
            "(randomStart:BOOLEAN), " +
            "(randomStartAfterSteps:INT), " +
            "(randomOrder:BOOLEAN), " +
            "(stepForward:BOOLEAN), " +
            "(stepSize:INT), " +
            "(shuffleSamples:BOOLEAN), " +
            "(sampleReverse:BOOLEAN), " +
            "(sampleSize:INT), " +
            "(cyclical:BOOLEAN)";

    /**
     * Input sample set for sampling.
     *
     */
    private final HashMap<Integer, MMatrix> inputs = new HashMap<>();

    /**
     * Output sample set for sampling.
     *
     */
    private final HashMap<Integer, MMatrix> outputs = new HashMap<>();

    /**
     * Input sample set for sampling.
     *
     */
    private final TreeSet<Integer> inputSampleSet = new TreeSet<>();

    /**
     * Number of validation cycles.
     *
     */
    private int numberOfIterations;

    /**
     * if true sampling takes place epoch wise i.e. samples are removed from sample set until all samples have been sampled at least once.
     *
     */
    private boolean perEpoch;

    /**
     * If true samples entire input as single set.
     *
     */
    private boolean fullSet;

    /**
     * If true samples at random start.
     *
     */
    private boolean randomStart;

    /**
     * If greater or equal to zero chooses random start after defined steps.
     *
     */
    private int randomStartAfterSteps;

    /**
     * Random start after steps count.
     *
     */
    private int randomStartAfterStepsCount = 0;

    /**
     * If true samples in random order.
     *
     */
    private boolean randomOrder;

    /**
     * If true sample steps in forward order (not valid for randomOrder sampling)-
     *
     */
    private boolean stepForward;

    /**
     * Number of steps taken forward or backward when sampling (no valid for randomOrder sampling).
     *
     */
    private int stepSize = 1;

    /**
     * If true shuffles sampled samples.
     *
     */
    private boolean shuffleSamples = false;

    /**
     * If true samples in reverse order (assumes no sample shuffling).
     *
     */
    private boolean sampleReverse = false;

    /**
     * Sampling size.
     *
     */
    private int sampleSize = 1;

    /**
     * If true considers input samples as cyclical.
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
     * Constructor for basic sampler.
     *
     * @param inputs input set for sampling.
     * @param outputs output set for sampling.
     * @throws NeuralNetworkException throws exception if input and output set sizes are not equal or not defined.
     */
    public BasicSampler(HashMap<Integer, MMatrix> inputs, HashMap<Integer, MMatrix> outputs) throws NeuralNetworkException {
        initializeDefaultParams();
        if (inputs == null || outputs == null) throw new NeuralNetworkException("Inputs or outputs are not defined.");
        if (inputs.isEmpty() || outputs.isEmpty()) throw new NeuralNetworkException("Input and output data sets cannot be empty.");
        if (inputs.size() != outputs.size()) throw new NeuralNetworkException("Size of sample inputs and outputs must match.");
        for (Map.Entry<Integer, MMatrix> entry : inputs.entrySet()) {
            int index = entry.getKey();
            MMatrix inputMMatrix = entry.getValue();
            addSample(inputMMatrix, outputs.get(index));
        }
        sampleAt = 0;
    }

    /**
     * Constructor for basic sampler.
     *
     * @param inputs input set for sampling.
     * @param outputs output set for sampling.
     * @param params parameters used for basic sampler.
     * @throws NeuralNetworkException throws exception if input and output set sizes are not equal or not defined.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public BasicSampler(HashMap<Integer, MMatrix> inputs, HashMap<Integer, MMatrix> outputs, String params) throws NeuralNetworkException, DynamicParamException {
        this(inputs, outputs);
        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        numberOfIterations = 1;
        perEpoch = false;
        fullSet = false;
        randomStart = false;
        randomOrder = true;
        randomStartAfterSteps = 0;
        stepForward = true;
        stepSize = 1;
        shuffleSamples = false;
        sampleReverse = false;
        sampleSize = 1;
        cyclical = false;
    }

    /**
     * Returns parameters used for basic sampler.
     *
     * @return parameters used for basic sampler.
     */
    public String getParamDefs() {
        return BasicSampler.paramNameTypes;
    }

    /**
     * Sets parameters used for basic sampler.<br>
     * <br>
     * Supported parameters are:<br>
     *     - numberOfIterations: number of training or validation iterations executed during step. Default value 1.<br>
     *     - perEpoch: if true sampling takes place epoch wise i.e. samples are removed from sample set until all samples have been sampled at least once. Default value false.<br>
     *     - fullSet: if true samples entire input as single set. Default value false.<br>
     *     - randomStart: if true samples at random start. Default value false.<br>
     *     - randomStartAfterSteps: if greater than zero chooses random start after defined steps. Default value -1.<br>
     *     - randomOrder: if true samples in random order. Default value true.<br>
     *     - stepForward: if true samples sampling steps in forward order (not valid for randomOrder sampling). Default value true.<br>
     *     - stepSize: number of steps taken forward or backward when sampling (not valid for randomOrder sampling). Default value 1.<br>
     *     - shuffleSamples: if true shuffles sampled samples. Default value false.<br>
     *     - sampleReverse: if true samples in reverse order (assumes no sample shuffling). Default value false.<br>
     *     - sampleSize: number of samples sampled. Default value 1.<br>
     *     - cyclical: if true considered sample set as cyclical. Default value false.<br>
     *
     * @param params parameters used for basic sampler.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("numberOfIterations")) {
            numberOfIterations = params.getValueAsInteger("numberOfIterations");
            if (numberOfIterations < 1) throw new DynamicParamException("Number of iterations must be at least 1.");
        }
        if (params.hasParam("perEpoch")) perEpoch = params.getValueAsBoolean("perEpoch");
        if (params.hasParam("fullSet")) fullSet = params.getValueAsBoolean("fullSet");
        if (params.hasParam("randomOrder")) randomOrder = params.getValueAsBoolean("randomOrder");
        if (params.hasParam("randomStart")) randomStart = params.getValueAsBoolean("randomStart");
        if (params.hasParam("randomStartAfterSteps")) randomStartAfterSteps = params.getValueAsInteger("randomStartAfterSteps");
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
        if (sampleDepth == -1) sampleDepth = input.getDepth();
        if (sampleDepth != input.getDepth() || sampleDepth != output.getDepth()) throw new NeuralNetworkException("All input and output samples must have same depth.");
        if (input.getDepth() != output.getDepth()) throw new NeuralNetworkException("Sample depth of input and output must match.");
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
     * @throws MatrixException throws exception if depth of sample is not matching depth of sequence.
     */
    public void getSamples(Sequence inputSequence, Sequence outputSequence) throws NeuralNetworkException, MatrixException {
        if (inputSequence.getDepth() != outputSequence.getDepth()) throw new NeuralNetworkException("Depth of samples input and output sequences must match");

        ArrayList<Integer> sampleIndices = getSampleIndices();

        for (Integer sampleIndex : sampleIndices) {
            inputSequence.put(sampleIndex, inputs.get(sampleIndex));
            outputSequence.put(sampleIndex, outputs.get(sampleIndex));
        }

    }

    /**
     * Returns sampled indices following sampling rules.
     *
     * @return sampled indices.
     */
    private ArrayList<Integer> getSampleIndices() {
        if (inputSampleSet.isEmpty()) inputSampleSet.addAll(inputs.keySet());

        ArrayList<Integer> sampleIndices = new ArrayList<>();
        if (randomOrder) {
            ArrayList<Integer> inputSamples = new ArrayList<>(inputSampleSet);
            Collections.shuffle(inputSamples);
            int maxSampleAmount = Math.min(sampleSize, inputSamples.size());
            for (int sampleIndex = 0; sampleIndex < maxSampleAmount; sampleIndex++) {
                sampleIndices.add(inputSamples.get(sampleIndex));
            }
        }
        else {
            if (fullSet) sampleIndices.addAll(inputSampleSet);
            else {
                if (randomStart || (randomStartAfterSteps > 0 && ++randomStartAfterSteps >= randomStartAfterStepsCount)) {
                    if (cyclical) sampleAt = random.nextInt(inputSampleSet.size());
                    else {
                        if (!stepForward) sampleAt = sampleSize + random.nextInt(inputSampleSet.size() - sampleSize);
                        else sampleAt = random.nextInt(Math.max(1, inputSampleSet.size() - sampleSize));
                    }
                }
                if (randomStartAfterSteps > 0 && randomStartAfterStepsCount > randomStartAfterSteps) randomStartAfterStepsCount = 0;
                int maxSampleAmount = Math.min(sampleSize, inputSampleSet.size());
                for (int sampleCount = 0; sampleCount < maxSampleAmount; sampleCount++) {
                    sampleIndices.add(sampleAt);
                    sampleAt += stepForward ? stepSize : -stepSize;
                    if (cyclical) {
                        if (stepForward) {
                            if (sampleAt > inputSampleSet.size() - 1)  sampleAt = sampleAt - (inputSampleSet.size() - 1);
                        }
                        else {
                            if (sampleAt < 0)  sampleAt = sampleAt + (inputSampleSet.size() - 1);
                        }
                    }
                    else {
                        if (stepForward) {
                            if (sampleAt > inputSampleSet.size() - 1)  {
                                sampleAt = 0;
                                break;
                            }
                        }
                        else {
                            if (sampleAt < 0)  {
                                sampleAt = inputSampleSet.size() - 1;
                                break;
                            }
                        }
                    }
                }
            }

            if (shuffleSamples) Collections.shuffle(sampleIndices);
            else if (sampleReverse) Collections.reverse(sampleIndices);

            if (perEpoch) for (Integer sampleIndex : sampleIndices) inputSampleSet.remove(sampleIndex);
        }

        return sampleIndices;
    }

}
