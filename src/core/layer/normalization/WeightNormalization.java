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
import utils.procedure.Procedure;
import utils.procedure.ProcedureFactory;

import java.util.HashMap;
import java.util.HashSet;

/**
 * Defines class for weight normalization.
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
     * Constructor for batch normalization layer.
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function for weight.
     * @param params parameters for feedforward layer.
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
        gMatrix = new DMatrix(g, "g");
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
     * Sets parameters used for weight Normalization.<br>
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
        if (params.hasParam("g")) g = params.getValueAsInteger("g");
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
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public MMatrix getInputMatrices(boolean resetPreviousInput) {
        return new MMatrix(input);
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
            Procedure procedure = new ProcedureFactory().getProcedure(this, getConstantMatrices());
            procedure.setStopGradient(getStopGradients(), true);
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
        HashSet<Matrix> stopGradients = new HashSet<>();
        stopGradients.add(gMatrix);
        return stopGradients;
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    protected HashSet<Matrix> getConstantMatrices() {
        HashSet<Matrix> constantMatrices = new HashSet<>();
        constantMatrices.add(input);
        constantMatrices.add(gMatrix);
        return constantMatrices;
    }

    /**
     * Takes single forward processing step to process layer input(s).<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardProcess() throws MatrixException, DynamicParamException {
        resetLayerOutputs();
        getLayerOutputs().putAll(getPreviousLayerOutputs());

        if (isTraining()) {
            for (Matrix weight : procedures.keySet()) {
                weights.put(weight, weight.copy());
                Procedure procedure = procedures.get(weight);
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
        resetLayerGradients();
        getLayerGradients().putAll(getNextLayerGradients());

        HashMap<Matrix, Matrix> nextLayerWeightGradients = getNextLayer().getLayerWeightGradients();
        for (Matrix weight : procedures.keySet()) {
            weight.setEqualTo(weights.get(weight));
            Matrix weightGradient = nextLayerWeightGradients.get(weight);
            Procedure procedure = procedures.get(weight);
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
     * Cumulates error from (L1 / L2 / Lp) regularization.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return cumulated error from regularization.
     */
    public double error() throws MatrixException, DynamicParamException {
        return getPreviousLayer() != null ? getPreviousLayer().error() : 0;
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
        System.out.println(getLayerName() + ": ");
        procedures.get(input).printGradientChain();
        System.out.println();
    }


}
