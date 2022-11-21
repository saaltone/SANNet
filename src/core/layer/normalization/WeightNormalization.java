/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.normalization;

import core.layer.AbstractExecutionLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.procedure.Procedure;
import utils.procedure.ProcedureFactory;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implements layer for weight normalization.
 *
 */
public class WeightNormalization extends AbstractExecutionLayer {

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
    private final transient HashMap<Matrix, Matrix> weights = new HashMap<>();

    /**
     * Weight normalization scalar.
     *
     */
    private double g;

    /**
     * Matrix for g value.
     *
     */
    private Matrix gMatrix;

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
     * Constructor for weight normalization layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for weight normalization layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public WeightNormalization(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        g = 1;
        gMatrix = new DMatrix(g);
        gMatrix.setName("g");
        gMatrix.setValue(0, 0, g);
    }

    /**
     * Returns parameters used for weight normalization layer.
     *
     * @return parameters used for weight normalization layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + WeightNormalization.paramNameTypes;
    }

    /**
     * Sets parameters used for weight normalization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - g: g multiplier value for normalization. Default value 1.<br>
     *
     * @param params parameters used for weight normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("g")) {
            g = params.getValueAsInteger("g");
            gMatrix.setValue(0, 0, g);
        }
    }

    /**
     * Checks if layer is recurrent layer type.
     *
     * @return always false.
     */
    public boolean isRecurrentLayer() { return false; }

    /**
     * Checks if layer works with recurrent layers.
     *
     * @return if true layer works with recurrent layers otherwise false.
     */
    public boolean worksWithRecurrentLayer() {
        return true;
    }

    /**
     * Returns reversed procedure.
     *
     * @return reversed procedure.
     */
    protected Procedure getReverseProcedure() {
        return null;
    }

    /**
     * Returns weight set.
     *
     * @return weight set.
     */
    protected WeightSet getWeightSet() {
        return null;
    }

    /**
     * Initializes neural network layer weights.
     *
     */
    public void initializeWeights() {
    }

    /**
     * Returns true if input is joined otherwise returns false.
     *
     * @return true if input is joined otherwise returns false.
     */
    protected boolean isJoinedInput() {
        return false;
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     * @throws MatrixException throws exception if matrix is exceeding its depth or matrix is not defined.
     */
    public TreeMap<Integer, MMatrix> getInputMatrices(boolean resetPreviousInput) throws MatrixException {
        return new TreeMap<>() {{ put(0, new MMatrix(input)); }};
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    protected void defineProcedure() throws MatrixException, DynamicParamException, NeuralNetworkException {
        if (getNextLayer().getWeightsMap().isEmpty()) throw new NeuralNetworkException("Unable initialize weight normalization. Next layer does not contain any weights.");

        procedures = new HashMap<>();
        HashSet<Matrix> nextLayerNormalizedWeights = getNextLayer().getNormalizedWeights();
        for (Matrix weight : nextLayerNormalizedWeights) {
            input = weight;
            Procedure procedure = new ProcedureFactory().getProcedure(this, null, getConstantMatrices(), getStopGradients(), null, isJoinedInput());
            procedures.put(weight, procedure);
        }
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
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    protected HashSet<Matrix> getStopGradients() {
        return new HashSet<>() {{ add(gMatrix); }};
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    protected HashSet<Matrix> getConstantMatrices() {
        return new HashSet<>() {{ add(input); add(gMatrix); }};
    }

    /**
     * Returns number of truncated steps for gradient calculation. -1 means no truncation.
     *
     * @return number of truncated steps.
     */
    protected int getTruncateSteps() {
        return -1;
    }

    /**
     * Takes single forward processing step to process layer input(s).<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardProcess() throws MatrixException, DynamicParamException {
        setLayerOutputs(getPreviousLayerOutputs());

        if (isTraining()) {
            for (Map.Entry<Matrix, Procedure> entry : procedures.entrySet()) {
                Matrix weight = entry.getKey();
                Procedure procedure = entry.getValue();
                weights.put(weight, weight.copy());
                procedure.reset();
                weight.setEqualTo(procedure.calculateExpression(weight, 0));
            }
        }
    }

    /**
     * Takes single backward processing step to process layer output gradient(s) towards input.<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void backwardProcess() throws MatrixException, DynamicParamException {
        setLayerGradients(getNextLayerGradients());

        HashMap<Matrix, Matrix> nextLayerWeightGradients = getNextLayer().getLayerWeightGradients();
        for (Map.Entry<Matrix, Procedure> entry : procedures.entrySet()) {
            Matrix weight = entry.getKey();
            Procedure procedure = entry.getValue();
            weight.setEqualTo(weights.get(weight));
            Matrix weightGradient = nextLayerWeightGradients.get(weight);
            weightGradient.setEqualTo(procedure.calculateGradient(weightGradient, 0));
        }
        weights.clear();
    }

    /**
     * Executes weight updates with optimizer.
     *
     */
    public void optimize() {
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "g value: " + g;
    }

    /**
     * Prints expression chains of normalization.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void printExpressions() throws NeuralNetworkException {
        if (procedures.size() == 0) return;
        System.out.println(getLayerName() + ": ");
        procedures.get(input).printExpressionChain();
        System.out.println();
    }

    /**
     * Prints gradient chains of normalization.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void printGradients() throws NeuralNetworkException {
        if (procedures.size() == 0) return;
        System.out.println(getLayerName() + ": ");
        procedures.get(input).printGradientChain();
        System.out.println();
    }


}
