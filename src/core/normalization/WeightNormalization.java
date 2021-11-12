/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.normalization;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.procedure.node.Node;
import utils.procedure.Procedure;
import utils.procedure.ProcedureFactory;

import java.util.HashMap;
import java.util.HashSet;

/**
 * Class that implements weight normalization for neural network layer.<br>
 * <br>
 * Reference: https://arxiv.org/pdf/1602.07868.pdf <br>
 *
 */
public class WeightNormalization extends AbstractNormalization {

    /**
     * Parameter name types for weight normalization.
     *     - g: g multiplier value for normalization. Default value 1.<br>
     *
     */
    private final static String paramNameTypes = "(g:INT)";

    /**
     * Map for un-normalized weights.
     *
     */
    private transient HashMap<Matrix, Matrix> weights = new HashMap<>();

    /**
     * Weight normalization scalar.
     *
     */
    private double g;

    /**
     * Matrix for g value.
     *
     */
    private final Matrix gMatrix = new DMatrix(g, "g");

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Procedures for weight normalization.
     *
     */
    private HashMap<Matrix, Procedure> procedures = new HashMap<>();

    /**
     * Constructor for weight normalization class.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public WeightNormalization() throws DynamicParamException {
        super(NormalizationType.WEIGHT_NORMALIZATION, WeightNormalization.paramNameTypes);
    }

    /**
     * Constructor for weight normalization class.
     *
     * @param params parameters for weight normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public WeightNormalization(String params) throws DynamicParamException {
        super(NormalizationType.WEIGHT_NORMALIZATION, WeightNormalization.paramNameTypes, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        g = 1;
    }

    /**
     * Sets parameters used for weight Normalization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - g: g multiplier value for normalization. Default value 1.<br>
     *
     * @param params parameters used for weight normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("g")) g = params.getValueAsInteger("g");
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
        MMatrix outputs = new MMatrix(1, "Output");
        Matrix output = input.multiply(gMatrix).divide(input.normAsMatrix(2));
        output.setName("Output");
        outputs.put(0, output);
        return outputs;
    }

    /**
     * Resets normalizer.
     *
     */
    public void reset() {
        weights = new HashMap<>();
    }

    /**
     * Reinitializes normalizer.
     *
     */
    public void reinitialize() {
        reset();
    }

    /**
     * Initializes normalization.
     *
     * @param node node for normalization.
     */
    public void initialize(Node node) {
    }

    /**
     * Initializes normalization.
     *
     * @param weight weight for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void initialize(Matrix weight) throws MatrixException, DynamicParamException {
        initializeProcedure(weight);
    }

    /**
     * Initializes weight normalization procedure.
     *
     * @param weight weight matrix for initialization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void initializeProcedure(Matrix weight) throws MatrixException, DynamicParamException {
        if (procedures == null) procedures = new HashMap<>();
        if (!procedures.containsKey(weight)) {
            input = weight;
            HashSet<Matrix> constantMatrices = new HashSet<>();
            constantMatrices.add(gMatrix);
            Procedure procedure = new ProcedureFactory().getProcedure(this, constantMatrices);
            procedure.setStopGradient(gMatrix, true);
            procedures.put(weight, procedure);
        }
    }

    /**
     * Normalizes each weight for forward step i.e. multiplies each weight matrix by g / sqrt(2-norm of weights).
     *
     * @param weight weight for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forward(Matrix weight) throws MatrixException, DynamicParamException {
        if (isTraining()) {
            weights.put(weight, weight.copy());
            procedures.get(weight).reset();
            weight.setEqualTo(procedures.get(weight).calculateExpression(weight, 0));
        }
    }

    /**
     * Finalizes forward step for normalization.<br>
     * Used typically for weight normalization.<br>
     *
     * @param weight weight for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardFinalize(Matrix weight) throws MatrixException {
        if (isTraining()) weight.setEqualTo(weights.get(weight));
    }

    /**
     * Executes backward propagation step for weight normalization.<br>
     * Calculates gradients backwards at step end for previous layer.<br>
     *
     * @param weight weight for backward normalization.
     * @param weightGradient gradient of weight for backward normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void backward(Matrix weight, Matrix weightGradient) throws MatrixException, DynamicParamException {
        weightGradient.setEqualTo(procedures.get(weight).calculateGradient(weightGradient, 0));
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
     * Executes optimizer step for normalizer.
     *
     */
    public void optimize() {}

    /**
     * Prints expression chains of normalization.
     *
     */
    public void printExpressions() {
        if (procedures.size() == 0) return;
        System.out.println("Normalization: " + getName() + ":");
        for (Procedure procedure : procedures.values()) procedure.printExpressionChain();
        System.out.println();
    }

    /**
     * Prints gradient chains of normalization.
     *
     */
    public void printGradients() {
        if (procedures.size() == 0) return;
        System.out.println("Normalization: " + getName() + ":");
        for (Procedure procedure : procedures.values()) procedure.printGradientChain();
        System.out.println();
    }

}