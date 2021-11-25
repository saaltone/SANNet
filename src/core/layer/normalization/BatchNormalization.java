/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer.normalization;

import core.layer.AbstractExecutionLayer;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.procedure.node.Node;
import utils.sampling.Sequence;

import java.util.HashSet;
import java.util.Random;

/**
 * Defines class for batch normalization.
 *
 */
public class BatchNormalization extends AbstractExecutionLayer {

    /**
     * Parameter name types for batch normalization.
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value false).<br>
     *     - momentum: degree of weighting decrease for exponential moving average. (default value 0.95).<br>
     *
     */
    private final static String paramNameTypes = "(meanOnly:BOOLEAN), " +
            "(momentum:DOUBLE)";

    /**
     * True if layer normalization is used calculation only with mean and variance excluded.
     *
     */
    private boolean meanOnly;

    /**
     * Input matrix for procedure construction.
     *
     */
    private MMatrix input;

    /**
     * Learnable parameter gamma of batch normalization layer.
     *
     */
    private Matrix gamma;

    /**
     * Learnable parameter beta of batch normalization layer.
     *
     */
    private Matrix beta;

    /**
     * Matrix for epsilon value.
     *
     */
    private Matrix epsilonMatrix;

    /**
     * Stores rolling average mean of samples when neural network is in training mode.<br>
     * Stored rolling average mean is used for normalization when neural network is in inference mode.<br>
     *
     */
    private Matrix averageMean;

    /**
     * Mean reference matrix.
     *
     */
    private transient Matrix mean;

    /**
     * Node for mean.
     *
     */
    private Node meanNode;

    /**
     * Stores rolling average variance of samples when neural network is in training mode.<br>
     * Stored rolling average variance is used for normalization when neural network is in inference mode.<br>
     *
     */
    private Matrix averageVariance;

    /**
     * Variance reference matrix.
     *
     */
    private transient Matrix variance;

    /**
     * Node for variance.
     *
     */
    private Node varianceNode;

    /**
     * Sample size for a batch. Double is used for calculation purposes.
     *
     */
    private double batchSize;

    /**
     * Momentum value for exponential moving average.
     *
     */
    private double momentum;

    /**
     * Constructor for batch normalization layer.
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function for weight.
     * @param params parameters for feedforward layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public BatchNormalization(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        epsilonMatrix = new DMatrix(10E-8);
        meanOnly = false;
        batchSize = -1;
        momentum = 0.95;
    }

    /**
     * Returns parameters used for batch normalization layer.
     *
     * @return parameters used for batch normalization layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + BatchNormalization.paramNameTypes;
    }

    /**
     * Sets parameters used for batch Normalization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value).<br>
     *     - momentum: degree of weighting decrease for exponential moving average. (default value 0.95).<br>
     *
     * @param params parameters used for layer normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("meanOnly")) meanOnly = params.getValueAsBoolean("meanOnly");
        if (params.hasParam("momentum")) momentum = params.getValueAsDouble("momentum");
    }

    /**
     * Checks if layer is recurrent layer type.
     *
     * @return always false.
     */
    public boolean isRecurrentLayer() { return false; }

    /**
     * Checks if layer is convolutional layer type.
     *
     * @return always false.
     */
    public boolean isConvolutionalLayer() { return false; }

    /**
     * Initializes batch normalization.<br>
     * Initializes weight and bias and their gradients.<br>
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void initialize() throws NeuralNetworkException, MatrixException, DynamicParamException {
        super.initialize();

        gamma = new DMatrix(getPreviousLayerWidth(), getPreviousLayerHeight(), (
row, col) -> new Random().nextGaussian() * 0.1, "Gamma");

        beta = new DMatrix(getPreviousLayerWidth(), getPreviousLayerHeight(), "Beta");

        registerWeight(gamma, false, false);

        registerWeight(beta, false, false);
    }

    /**
     * Reinitializes layer.
     *
     */
    public void reinitialize() {
        gamma.initialize((row, col) -> new Random().nextGaussian() * 0.1);
        beta.reset();
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public MMatrix getInputMatrices(boolean resetPreviousInput) throws MatrixException {
        input = new MMatrix(1, "Inputs");
        input.put(0, new DMatrix(getPreviousLayerWidth(), getPreviousLayerHeight(), Initialization.ONE, "Inputs"));
        return input;
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    protected void defineProcedure() throws MatrixException, DynamicParamException, NeuralNetworkException {
        super.defineProcedure();

        meanNode = procedure.getNode(mean);
        varianceNode = procedure.getNode(variance);
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException, DynamicParamException {
        mean = input.mean();
        MMatrix meanNormalizedInput = input.subtract(mean);

        MMatrix outputs;
        if (!meanOnly) {
            variance = input.variance(mean);
            MMatrix normalizedOutput = meanNormalizedInput.divide(variance.add(epsilonMatrix).apply(UnaryFunctionType.SQRT));
            outputs = normalizedOutput.multiply(gamma).add(beta);
        }
        else outputs = meanNormalizedInput.multiply(gamma).add(beta);
        outputs.setName("Outputs", true);

        return outputs;
    }

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    protected HashSet<Matrix> getStopGradients() {
        HashSet<Matrix> stopGradients = new HashSet<>();
        stopGradients.add(epsilonMatrix);
        return stopGradients;
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    protected HashSet<Matrix> getConstantMatrices() {
        HashSet<Matrix> constantMatrices = new HashSet<>();
        constantMatrices.add(epsilonMatrix);
        return constantMatrices;
    }

    /**
     * Takes single forward processing step to process layer input(s).<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardProcess() throws MatrixException, DynamicParamException {
        Sequence inputSequence = prepareForwardProcess();
        if (isTraining()) {
            if (batchSize == -1) batchSize = inputSequence.sampleSize();
            if (inputSequence.sampleSize() < 2) throw new MatrixException("Batch normalization must have minimum batch size of 2 for training phase.");

            procedure.calculateExpression(inputSequence, getLayerOutputs());

            averageMean = meanNode.getMatrix().exponentialMovingAverage(averageMean, momentum);
            if (!meanOnly) averageVariance = varianceNode.getMatrix().exponentialMovingAverage(averageVariance, momentum);
        }
        else {
            if (!meanOnly) {
                Matrix averageStandardDeviation = averageVariance.multiply(batchSize / (batchSize - 1)).add(epsilonMatrix).apply(UnaryFunctionType.SQRT);
                for (Integer sampleIndex : inputSequence.keySet()) {
                    for (Integer entryIndex : inputSequence.get(sampleIndex).keySet()) {
                        getLayerOutputs().put(sampleIndex, entryIndex, inputSequence.get(sampleIndex, entryIndex).subtract(averageMean).divide(averageStandardDeviation).multiply(gamma).add(beta));
                    }
                }
            }
            else {
                for (Integer sampleIndex : inputSequence.keySet()) {
                    for (Integer entryIndex : inputSequence.get(sampleIndex).keySet()) {
                        getLayerOutputs().put(sampleIndex, entryIndex, inputSequence.get(sampleIndex, entryIndex).subtract(averageMean).multiply(gamma).add(beta));
                    }
                }
            }
        }
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "Mean only: " + meanOnly + ", Momentum: " + momentum;
    }

}
