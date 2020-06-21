/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.normalization;

import core.optimization.Optimizer;
import utils.*;
import utils.matrix.*;
import utils.procedure.ForwardProcedure;
import utils.procedure.Node;
import utils.procedure.Procedure;
import utils.procedure.ProcedureFactory;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

/**
 * Class that implements Layer Normalization for neural network layer.<br>
 * Layer Normalization is particularly well suited for recurrent neural networks as it normalizes each sample independently.<br>
 * <br>
 * Reference: https://www.cs.toronto.edu/~hinton/absps/LayerNormalization.pdf<br>
 *
 */
public class LayerNormalization implements Normalization, ForwardProcedure, Serializable {

    private static final long serialVersionUID = 3466341546851269706L;

    /**
     * Type of normalization.
     *
     */
    private final NormalizationType normalizationType;

    /**
     * True if layer normalization is used calculation only with mean and variance excluded.
     *
     */
    private boolean meanOnly = false;

    /**
     * Optimizer for Layer normalization;
     *
     */
    private Optimizer optimizer;

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Learnable parameter gamma of Layer normalization layer.
     *
     */
    private Matrix gamma;

    /**
     * Learnable parameter beta of Layer normalization layer.
     *
     */
    private Matrix beta;

    /**
     * Set of weights to be managed.
     *
     */
    private final HashSet<Matrix> weights = new HashSet<>();

    /**
     * Set of weights to be managed.
     *
     */
    private HashMap<Matrix, Matrix> weightGradients = new HashMap<>();

    /**
     * Procedure for layer normalization.
     *
     */
    private Procedure procedure;

    /**
     * Matrix for epsilon value.
     *
     */
    private final Matrix epsilonMatrix = new DMatrix(10E-8);

    /**
     * Constructor for Layer normalization class.
     *
     * @param normalizationType normalizationType.
     */
    public LayerNormalization(NormalizationType normalizationType) {
        this.normalizationType = normalizationType;
    }

    /**
     * Constructor for Layer normalization class.
     *
     * @param normalizationType normalizationType.
     * @param params parameters for Layer normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public LayerNormalization(NormalizationType normalizationType, String params) throws DynamicParamException {
        this(normalizationType);
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for Layer normalization.
     *
     * @return parameters used for Layer normalization.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("meanOnly", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for Layer normalization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value).<br>
     *
     * @param params parameters used for Layer normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("meanOnly")) meanOnly = params.getValueAsBoolean("meanOnly");
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public MMatrix getInputMatrices(boolean resetPreviousInput) {
        return new MMatrix(input);
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException {
        Matrix meanNormalizedInput = input.subtract(input.meanAsMatrix());

        Matrix output;
        if (!meanOnly) {
            Matrix normalizedOutput = meanNormalizedInput.divide(input.varianceAsMatrix().add(epsilonMatrix).apply(UnaryFunctionType.SQRT));
            output = normalizedOutput.multiply(gamma).add(beta);
        }
        else output =  meanNormalizedInput.multiply(gamma).add(beta);
        output.setName("Output");
        MMatrix outputs = new MMatrix(1, "Output");
        outputs.put(0, output);

        return outputs;
    }

    /**
     * Resets Layer normalizer.
     *
     */
    public void reset() {
        weightGradients = new HashMap<>();
    }

    /**
     * Sets flag for Layer normalization if neural network is in training state.
     *
     * @param isTraining if true neural network is in state otherwise false.
     */
    public void setTraining(boolean isTraining) {
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
     */
    public void initialize(Node node) throws MatrixException {
        initializeProcedure(node.getMatrix(node.firstKey()));
    }

    /**
     * Initializes normalization.
     *
     * @param weight weight for normalization.
     */
    public void initialize(Matrix weight) {
    }

    /**
     * Initializes layer normalization procedure.
     *
     * @param inputMatrix input matrix for initialization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private void initializeProcedure(Matrix inputMatrix) throws MatrixException {
        int rows = inputMatrix.getRows();
        int columns = inputMatrix.getColumns();

        input = new DMatrix(rows, columns, Initialization.ONE, "Input");

        gamma = new DMatrix(rows, columns, (row, col) -> new Random().nextGaussian() * 0.1, "gamma");
        weights.add(gamma);

        beta = new DMatrix(rows, columns, "beta");
        weights.add(beta);

        procedure = new ProcedureFactory().getProcedure(this, weights);
    }

    /**
     * Executes forward propagation step for Layer normalization at step start.<br>
     * Calculates feature wise mean and variance for each sample independently.<br>
     * Removes mean and variance from input samples.<br>
     *
     * @param node node for normalization.
     * @param inputIndex input index for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forward(Node node, int inputIndex) throws MatrixException {
        Matrix inputMatrix = node.getMatrix(inputIndex);
        if (inputMatrix.size() < 2) return;
        procedure.reset();
        node.setMatrix(inputIndex, procedure.calculateExpression(inputMatrix));
    }

    /**
     * Executes backward propagation step for Layer normalization.<br>
     * Calculates gradients backwards at step end for previous layer.<br>
     *
     * @param node node for normalization.
     * @param outputIndex input index for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backward(Node node, int outputIndex) throws MatrixException {
        Matrix outputGradient =node.getGradient(outputIndex);
        if (outputGradient.size() < 2) return;

        node.setGradient(outputIndex, procedure.calculateGradient(outputGradient));

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
     */
    public void forward(Node node) {}

    /**
     * Not used.
     *
     * @param node node for normalization.
     */
    public void backward(Node node) {}

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
     * @param weight weight for backward's normalization.
     * @param weightGradient gradient of weight for backward normalization.
     */
    public void backward(Matrix weight, Matrix weightGradient) {
    }

    /**
     * Executes optimizer step for normalizer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void optimize() throws MatrixException {
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
        System.out.print("Normalization: " + getName() + ": ");
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