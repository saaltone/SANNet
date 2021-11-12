/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.normalization;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.procedure.Procedure;
import utils.procedure.ProcedureFactory;
import utils.procedure.node.Node;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

/**
 * Class that implements batch normalization for neural network layer.<br>
 * <br>
 * Reference: http://proceedings.mlr.press/v37/ioffe15.pdf <br>
 *
 */
public class BatchNormalization extends AbstractNormalization {

    /**
     * Parameter name types for batch normalization.
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value).<br>
     *     - beta: degree of weighting decrease for exponential moving average. Default value 0.9.<br>
     *
     */
    private final static String paramNameTypes = "(meanOnly:BOOLEAN), " +
            "(beta:INT)";

    /**
     * Degree of weighting decrease for exponential moving average. Default value 0.9.
     *
     */
    private double betaValue;

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
     * Set of weights to be managed.
     *
     */
    private final HashSet<Matrix> weights = new HashSet<>();

    /**
     * Set of weight gradients to be managed.
     *
     */
    private HashMap<Matrix, Matrix> weightGradients = new HashMap<>();

    /**
     * Number of input rows.
     *
     */
    private transient int inputRows;

    /**
     * Number of input columns.
     *
     */
    private transient int inputColumns;

    /**
     * Input size.
     *
     */
    private transient int inputSize;

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
     * True if batch normalization is used calculation only with mean and variance excluded.
     *
     */
    private boolean meanOnly;

    /**
     * Input matrix for procedure construction.
     *
     */
    private MMatrix input;

    /**
     * Epsilon term for batch normalization. Default value 10E-8.<br>
     * Term provides mathematical stability for normalizer.<br>
     *
     */
    private final Matrix epsilonMatrix = new DMatrix(10E-8, "epsilon");

    /**
     * Default constructor for batch normalization class.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public BatchNormalization() throws DynamicParamException {
        super(NormalizationType.BATCH_NORMALIZATION, BatchNormalization.paramNameTypes);
    }

    /**
     * Constructor for batch normalization class.
     *
     * @param params parameters for batch normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public BatchNormalization(String params) throws DynamicParamException {
        super(NormalizationType.BATCH_NORMALIZATION, BatchNormalization.paramNameTypes, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        betaValue = 0.9;
        meanOnly = false;
    }

    /**
     * Sets parameters used for batch Normalization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value).<br>
     *     - beta: degree of weighting decrease for exponential moving average. Default value 0.9.<br>
     *
     * @param params parameters used for batch normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("meanOnly")) meanOnly = params.getValueAsBoolean("meanOnly");
        if (params.hasParam("beta")) betaValue = params.getValueAsDouble("beta");
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public MMatrix getInputMatrices(boolean resetPreviousInput) throws MatrixException {
        input = new MMatrix(inputSize, "Inputs");
        for (int index = 0; index < inputSize; index++) input.put(index, new DMatrix(inputRows, inputColumns, Initialization.ONE, "Inputs"));
        return input;
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
     * Initializes batch normalization procedure.
     *
     * @param node node for initialization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void initializeProcedure(Node node) throws MatrixException, DynamicParamException {
        if (input != null) return;

        inputRows = node.getEmptyMatrix().getRows();
        inputColumns = node.getEmptyMatrix().getColumns();
        inputSize = node.size();

        gamma = new DMatrix(inputRows, inputColumns, (row, col) -> new Random().nextGaussian() * 0.1, "gamma");
        weights.add(gamma);

        beta = new DMatrix(inputRows, inputColumns, "beta");
        weights.add(beta);

        HashSet<Matrix> constantMatrices = new HashSet<>(weights);
        constantMatrices.add(epsilonMatrix);
        Procedure procedure = new ProcedureFactory().getProcedure(this, constantMatrices);
        procedure.setStopGradient(epsilonMatrix, true);

        meanNode = procedure.getNode(mean);
        varianceNode = procedure.getNode(variance);

        setProcedure(procedure);
    }

    /**
     * Resets batch normalizer.
     *
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void reset() throws MatrixException {
        weightGradients = new HashMap<>();
        if (getProcedure() != null) getProcedure().reset();
    }

    /**
     * Reinitializes normalizer.
     *
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void reinitialize() throws MatrixException {
        if (gamma != null) gamma.initialize((row, col) -> new Random().nextGaussian() * 0.1);
        if (beta != null) beta.reset();
        reset();
    }

    /**
     * Initializes normalization.
     *
     * @param node node for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void initialize(Node node) throws MatrixException, DynamicParamException {
        initializeProcedure(node);
    }

    /**
     * Initializes normalization.
     *
     * @param weight weight for normalization.
     */
    public void initialize(Matrix weight) {
    }

    /**
     * Executes forward propagation step for batch normalization at step start.<br>
     * Calculates feature wise mean and variance for batch of samples.<br>
     * Stores mean and variance into rolling averages respectively.<br>
     * Removes mean and variance from input samples.<br>
     *
     * @param node node for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forward(Node node) throws MatrixException, DynamicParamException {
        if (isTraining()) {
            if (node.size() < 2) return;
            batchSize = node.size();

            node.setMatrices(getProcedure().calculateExpression(node.getMatrices()));

            averageMean = meanNode.getMatrix().exponentialMovingAverage(averageMean, betaValue);
            if (!meanOnly) averageVariance = varianceNode.getMatrix().exponentialMovingAverage(averageVariance, betaValue);
        }
        else {
            if (!meanOnly) {
                Matrix averageStandardDeviation = averageVariance.add(epsilonMatrix).multiply(batchSize / (batchSize - 1)).apply(UnaryFunctionType.SQRT);
                for (Integer sampleIndex : node.keySet()) {
                    node.setMatrix(sampleIndex, node.getMatrix(sampleIndex).subtract(averageMean).divide(averageStandardDeviation).multiply(gamma).add(beta));
                }
            }
            else {
                for (Integer sampleIndex : node.keySet()) {
                    node.setMatrix(sampleIndex, node.getMatrix(sampleIndex).subtract(averageMean).multiply(gamma).add(beta));
                }
            }
        }
    }

    /**
     * Executes backward propagation step for batch normalization.<br>
     * Calculates gradients backwards at step end for previous layer.<br>
     *
     * @param node node for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void backward(Node node) throws MatrixException, DynamicParamException {
        if (node.size() < 2 || !isTraining()) return;

        node.setGradients(getProcedure().calculateGradient(node.getGradients()));

        updateWeightGradient(gamma, weightGradients);

        updateWeightGradient(beta, weightGradients);
    }

    /**
     * Not used.
     *
     * @param node node for normalization.
     * @param inputIndex input index for normalization.
     */
    public void forward(Node node, int inputIndex) {}

    /**
     * Not used.
     *
     * @param node node for normalization.
     * @param outputIndex input index for normalization.
     */
    public void backward(Node node, int outputIndex) {}

    /**
     * Not used.
     *
     * @param weight weight for normalization.
     */
    public void forward(Matrix weight) {
    }

    /**
     * Not used.
     *
     * @param weight weight for normalization.
     */
    public void forwardFinalize(Matrix weight) {
    }

    /**
     * Not used.
     *
     * @param weight weight for backward normalization.
     * @param weightGradient gradient of weight for backward normalization.
     */
    public void backward(Matrix weight, Matrix weightGradient) {
    }

    /**
     * Executes optimizer step for normalizer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void optimize() throws MatrixException, DynamicParamException {
        weightGradients = optimize(weightGradients);
    }

}