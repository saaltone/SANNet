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

import java.util.HashSet;
import java.util.Random;

/**
 * Defines class for layer normalization.
 *
 */
public class LayerNormalization extends AbstractExecutionLayer {

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
     * Matrix for epsilon value.
     *
     */
    private Matrix epsilonMatrix;

    /**
     * Constructor for layer normalization layer.
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function for weight.
     * @param params parameters for feedforward layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public LayerNormalization(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
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
    }

    /**
     * Returns parameters used for layer normalization layer.
     *
     * @return parameters used for layer normalization layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + LayerNormalization.paramNameTypes;
    }

    /**
     * Sets parameters used for layer normalization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value false).<br>
     *
     * @param params parameters used for layer normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("meanOnly")) meanOnly = params.getValueAsBoolean("meanOnly");
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
     * Initializes layer normalization.<br>
     * Initializes weight and bias and their gradients.<br>
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void initialize() throws NeuralNetworkException, MatrixException, DynamicParamException {
        super.initialize();

        gamma = new DMatrix(getPreviousLayerWidth(), getPreviousLayerHeight(), (row, col) -> new Random().nextGaussian() * 0.1, "Gamma");

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
    public MMatrix getInputMatrices(boolean resetPreviousInput) {
        input = new DMatrix(getPreviousLayerWidth(), getPreviousLayerHeight(), Initialization.ONE, "Input");
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
        Matrix output = !meanOnly ? input.subtract(input.meanAsMatrix()).divide(input.varianceAsMatrix().add(epsilonMatrix).apply(UnaryFunctionType.SQRT)).multiply(gamma).add(beta) : input.subtract(input.meanAsMatrix()).multiply(gamma).add(beta);

        output.setName("Output");
        MMatrix outputs = new MMatrix(1, "Output");
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
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "Mean only: " + meanOnly;
    }

}
