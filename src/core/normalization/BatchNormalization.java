/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.normalization;

import core.optimization.Optimizer;
import utils.*;
import utils.matrix.*;
import utils.procedure.ForwardProcedure;
import utils.procedure.node.Node;
import utils.procedure.Procedure;
import utils.procedure.ProcedureFactory;

import java.io.Serializable;
import java.util.*;

/**
 * Class that implements Batch Normalization for neural network layer.<br>
 * <br>
 * Reference: http://proceedings.mlr.press/v37/ioffe15.pdf<br>
 *
 */
public class BatchNormalization implements Normalization, ForwardProcedure, Serializable {

    private static final long serialVersionUID = 3466341546851269706L;

    /**
     * Type of normalization.
     *
     */
    private final NormalizationType normalizationType;

    /**
     * Degree of weighting decrease for exponential moving average. Default value 0.9.
     *
     */
    private double betaValue = 0.9;

    /**
     * If true neural network is in state otherwise false.
     *
     */
    private transient boolean isTraining;

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
    private boolean meanOnly = false;

    /**
     * Optimizer for batch normalization;
     *
     */
    private Optimizer optimizer;

    /**
     * Input matrix for procedure construction.
     *
     */
    private MMatrix input;

    /**
     * Procedure for layer normalization.
     *
     */
    private Procedure procedure;

    /**
     * Epsilon term for batch normalization. Default value 10E-8.<br>
     * Term provides mathematical stability for normalizer.<br>
     *
     */
    private final Matrix epsilonMatrix = new DMatrix(10E-8, "epsilon");

    /**
     * Default constructor for batch normalization class.
     *
     * @param normalizationType normalizationType.
     */
    public BatchNormalization(NormalizationType normalizationType) {
        this.normalizationType = normalizationType;
    }

    /**
     * Constructor for batch normalization class.
     *
     * @param normalizationType normalizationType.
     * @param params parameters for batch normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public BatchNormalization(NormalizationType normalizationType, String params) throws DynamicParamException {
        this(normalizationType);
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for batch normalization.
     *
     * @return parameters used for batch normalization.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("meanOnly", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("beta", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
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
     * @param inputs input matrix for initialization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void initializeProcedure(MMatrix inputs) throws MatrixException, DynamicParamException {
        if (input != null) return;

        inputRows = inputs.get(inputs.firstKey()).getRows();
        inputColumns = inputs.get(inputs.firstKey()).getColumns();
        inputSize = inputs.size();

        gamma = new DMatrix(inputRows, inputColumns, (row, col) -> new Random().nextGaussian() * 0.1, "gamma");
        weights.add(gamma);

        beta = new DMatrix(inputRows, inputColumns, "beta");
        weights.add(beta);

        procedure = new ProcedureFactory().getProcedure(this, weights);

        meanNode = procedure.getNode(mean);
        varianceNode = procedure.getNode(variance);
    }

    /**
     * Resets batch normalizer.
     *
     */
    public void reset() {
        weightGradients = new HashMap<>();
        if (procedure != null) procedure.reset();
    }

    /**
     * Reinitializes normalizer.
     *
     */
    public void reinitialize() {
        if (gamma != null) gamma.initialize((row, col) -> new Random().nextGaussian() * 0.1);
        if (beta != null) beta.reset();
        reset();
    }

    /**
     * Sets flag for batch normalization if neural network is in training state.
     *
     * @param isTraining if true neural network is in state otherwise false.
     */
    public void setTraining(boolean isTraining) {
        this.isTraining = isTraining;
    }

    /**
     * Sets optimizer for normalizer.
     *
     * @param optimizer optimizer
     */
    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    /**
     * Initializes normalization.
     *
     * @param node node for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void initialize(Node node) throws MatrixException, DynamicParamException {
        initializeProcedure(node.getMatrices());
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
        if (node.size() < 2) return;

        if (isTraining) {
            batchSize = node.size();

            node.setMatrices(procedure.calculateExpression(node.getMatrices()));

            averageMean = Matrix.exponentialMovingAverage(averageMean, meanNode.getMatrix(), betaValue);
            if (!meanOnly) averageVariance = Matrix.exponentialMovingAverage(averageVariance, varianceNode.getMatrix(), betaValue);
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
        if (node.size() < 2) return;

        node.setGradients(procedure.calculateGradient(node.getGradients()));

        Matrix gammaGradient = procedure.getGradient(gamma);
        if (!weightGradients.containsKey(gamma)) weightGradients.put(gamma, gammaGradient);
        else weightGradients.get(gamma).add(gammaGradient, weightGradients.get(gamma));

        Matrix betaGradient = procedure.getGradient(beta);
        if (!weightGradients.containsKey(beta)) weightGradients.put(beta, betaGradient);
        else weightGradients.get(beta).add(betaGradient, weightGradients.get(beta));
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
        for (Matrix weight : weightGradients.keySet()) optimizer.optimize(weight, weightGradients.get(weight));
        weightGradients = new HashMap<>();
    }

    /**
     * Returns name of normalization.
     *
     * @return name of normalization.
     */
    public String getName() {
        return normalizationType.toString();
    }

    /**
     * Prints expression chains of normalization.
     *
     */
    public void printExpressions() {
        if (procedure == null) return;
        System.out.println("Normalization: " + getName() + ": ");
        procedure.printExpressionChain();
        System.out.println();
    }

    /**
     * Prints gradient chains of normalization.
     *
     */
    public void printGradients() {
        if (procedure == null) return;
        System.out.println("Normalization: " + getName() + ": ");
        procedure.printGradientChain();
        System.out.println();
    }

}