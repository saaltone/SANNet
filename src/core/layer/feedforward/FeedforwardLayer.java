/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.layer.feedforward;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.AbstractExecutionLayer;
import utils.*;
import utils.matrix.*;

import java.util.HashMap;

/**
 * Implements non-recurrent feed forward layer.
 *
 */
public class FeedforwardLayer extends AbstractExecutionLayer {

    /**
     * Weight matrix.
     *
     */
    private Matrix weight;

    /**
     * Bias matrix.
     *
     */
    private Matrix bias;

    /**
     * Activation function for feedforward layer.
     *
     */
    protected final ActivationFunction activationFunction;

    /**
     * True if weights are regulated otherwise weights are not regulated.
     *
     */
    private boolean regulateDirectWeights = true;

    /**
     * If true normalizes at input otherwise normalizes prior non-linearity.
     *
     */
    private boolean normalizeAtInput = false;

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Constructor for feed forward layer.
     *
     * @param layerIndex layer Index.
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight.
     * @param params parameters for feed forward layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public FeedforwardLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
        setParams(new DynamicParam(params, getParamDefs()));
        if (activationFunction != null) this.activationFunction = activationFunction;
        else this.activationFunction = new ActivationFunction(UnaryFunctionType.ELU);
    }

    /**
     * Returns parameters used for feed forward layer.
     *
     * @return parameters used for feed forward layer.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.put("regulateDirectWeights", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("normalizeAtInput", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for feed forward layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - regulateDirectWeights: true if (direct) weights are regulated otherwise false (default value).<br>
     *     - normalizeAtInput: if true normalizes at input otherwise normalizes prior non-linearity.
     *
     * @param params parameters used for feed forward layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
        if (params.hasParam("normalizeAtInput")) normalizeAtInput = params.getValueAsBoolean("normalizeAtInput");
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
     * Initializes feed forward layer.<br>
     * Initializes weight and bias and their gradients.<br>
     *
     */
    public void initialize() {
        int previousLayerWidth = getPreviousLayerWidth();
        int layerWidth = getLayerWidth();

        weight = new DMatrix(layerWidth, previousLayerWidth, this.initialization, "Weight");

        bias = new DMatrix(layerWidth, 1, "Bias");

        registerWeight(weight, regulateDirectWeights, true);

        registerWeight(bias, false, false);

    }

    /**
     * Reinitializes layer.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void reinitialize() throws MatrixException, NeuralNetworkException {
        weight.initialize(this.initialization);
        bias.reset();

        super.reinitialize();
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public MMatrix getInputMatrices(boolean resetPreviousInput) {
        input = new DMatrix(getPreviousLayerWidth(), 1, Initialization.ONE, "Input");
        return new MMatrix(input);
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException {
        if (normalizeAtInput) input.setNormalize(true);
        input.setRegularize(true);

        Matrix output = weight.dot(input);

        if (!normalizeAtInput) output.setNormalize(true);

        output = output.add(bias);

        output = output.apply(activationFunction);

        output.setName("Output");
        MMatrix outputs = new MMatrix(1, "Output");
        outputs.put(0, output);
        return outputs;

    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "Activation function: " + activationFunction.getName();
    }

}
