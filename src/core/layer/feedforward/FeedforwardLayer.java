/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer.feedforward;

import core.network.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.AbstractExecutionLayer;
import utils.*;
import utils.matrix.*;

import java.util.HashSet;

/**
 * Implements non-recurrent feedforward layer.<br>
 *
 */
public class FeedforwardLayer extends AbstractExecutionLayer {

    /**
     * Parameter name types for feedforward layer.
     *     - regulateDirectWeights: true if (direct) weights are regulated otherwise false. Default value true.<br>
     *     - normalizeAtInput: if true normalizes at input otherwise normalizes prior non-linearity. Default value false.<br>
     *     - splitOutputAtPosition: splits output at specific position.<br>
     *
     */
    private final static String paramNameTypes = "(regulateDirectWeights:BOOLEAN), " +
            "(normalizeAtInput:BOOLEAN), " +
            "(splitOutputAtPosition:INT)";

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
    private boolean regulateDirectWeights;

    /**
     * If true normalizes at input otherwise normalizes prior non-linearity.
     *
     */
    private boolean normalizeAtInput;

    /**
     * Splits output at given position.
     *
     */
    private int splitOutputAtPosition;

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Constructor for feedforward layer.
     *
     * @param layerIndex layer Index.
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight.
     * @param params parameters for feedforward layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public FeedforwardLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
        this.activationFunction = activationFunction != null ? activationFunction : new ActivationFunction(UnaryFunctionType.RELU);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        regulateDirectWeights = true;
        normalizeAtInput = false;
        splitOutputAtPosition = -1;
    }

    /**
     * Returns parameters used for feedforward layer.
     *
     * @return parameters used for feedforward layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + FeedforwardLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for feedforward layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - regulateDirectWeights: true if (direct) weights are regulated otherwise false. Default value true.<br>
     *     - normalizeAtInput: if true normalizes at input otherwise normalizes prior non-linearity. Default value false.<br>
     *     - splitOutputAtPosition: splits output at specific position.<br>
     *
     * @param params parameters used for feedforward layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
        if (params.hasParam("normalizeAtInput")) normalizeAtInput = params.getValueAsBoolean("normalizeAtInput");
        if (params.hasParam("splitOutputAtPosition")) splitOutputAtPosition = params.getValueAsInteger("splitOutputAtPosition");
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
     * Initializes feedforward layer.<br>
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

        if (splitOutputAtPosition == -1) output = output.apply(activationFunction);
        else {
            Matrix result = output.split(splitOutputAtPosition, true);
            output.apply(result, activationFunction);
            output = result;
        }

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
        return new HashSet<>();
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
