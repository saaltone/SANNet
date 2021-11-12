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
 * Class that implements layer normalization for neural network layer.<br>
 * Layer normalization is particularly well suited for recurrent neural networks as it normalizes each sample independently.<br>
 * <br>
 * Reference: https://www.cs.toronto.edu/~hinton/absps/LayerNormalization.pdf <br>
 *
 */
public class LayerNormalization extends AbstractNormalization {

    /**
     * Parameter name types for layer normalization.
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value false).<br>
     *
     */
    private final static String paramNameTypes = "(meanOnly:BOOLEAN)";

    /**
     * True if layer normalization is used calculation only with mean and variance excluded.
     *
     */
    private boolean meanOnly;

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Learnable parameter gamma of layer normalization layer.
     *
     */
    private Matrix gamma;

    /**
     * Learnable parameter beta of layer normalization layer.
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
     * Matrix for epsilon value.
     *
     */
    private final Matrix epsilonMatrix = new DMatrix(10E-8);

    /**
     * Constructor for layer normalization class.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public LayerNormalization() throws DynamicParamException {
        super(NormalizationType.LAYER_NORMALIZATION, LayerNormalization.paramNameTypes);
    }

    /**
     * Constructor for layer normalization class.
     *
     * @param params parameters for layer normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public LayerNormalization(String params) throws DynamicParamException {
        super(NormalizationType.LAYER_NORMALIZATION, LayerNormalization.paramNameTypes, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        meanOnly = false;
    }

    /**
     * Sets parameters used for layer normalization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value false).<br>
     *
     * @param params parameters used for layer normalization.
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
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException, DynamicParamException {
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
     * Resets layer normalizer.
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
        initializeProcedure(node.getEmptyMatrix());
//        initializeProcedure(node.getMatrix(node.firstKey()));
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
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void initializeProcedure(Matrix inputMatrix) throws MatrixException, DynamicParamException {
        if (input != null) return;

        int rows = inputMatrix.getRows();
        int columns = inputMatrix.getColumns();

        input = new DMatrix(rows, columns, Initialization.ONE, "Input");

        gamma = new DMatrix(rows, columns, (row, col) -> new Random().nextGaussian() * 0.1, "gamma");
        weights.add(gamma);

        beta = new DMatrix(rows, columns, "beta");
        weights.add(beta);

        HashSet<Matrix> constantMatrices = new HashSet<>(weights);
        constantMatrices.add(epsilonMatrix);
        Procedure procedure = new ProcedureFactory().getProcedure(this, constantMatrices);
        procedure.setStopGradient(epsilonMatrix, true);

        setProcedure(procedure);
    }

    /**
     * Executes forward propagation step for Layer normalization at step start.<br>
     * Calculates feature wise mean and variance for each sample independently.<br>
     * Removes mean and variance from input samples.<br>
     *
     * @param node node for normalization.
     * @param inputIndex input index for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forward(Node node, int inputIndex) throws MatrixException, DynamicParamException {
        Matrix inputMatrix = node.getMatrix(inputIndex);
        if (inputMatrix.size() < 2) return;
        node.setMatrix(inputIndex, getProcedure().calculateExpression(inputMatrix, inputIndex));
    }

    /**
     * Executes backward propagation step for layer normalization.<br>
     * Calculates gradients backwards at step end for previous layer.<br>
     *
     * @param node node for normalization.
     * @param outputIndex input index for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void backward(Node node, int outputIndex) throws MatrixException, DynamicParamException {
        Matrix outputGradient = node.getGradient(outputIndex);
        if (outputGradient.size() < 2) return;

        node.setGradient(outputIndex, getProcedure().calculateGradient(outputGradient, outputIndex));

        updateWeightGradient(gamma, weightGradients);

        updateWeightGradient(beta, weightGradients);
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